import streamlit as st
import cv2
import subprocess
import pandas as pd
import numpy as np
import time
from datetime import datetime
from pathlib import Path
import urllib.parse
import altair as alt

# =============================================================================
# Traffic Data Quality Dashboard: Real-Time & Static with Fixed Axes
# =============================================================================

# Page config
st.set_page_config(page_title="Traffic Data Quality Dashboard", layout="wide")

# Directories
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "video_data"
DATA_DIR.mkdir(exist_ok=True)

# Predefined YouTube sources
VIDEO_LIST = [
    {"id": "cam01", "name": "4 Corners Downtown", "yt_link": "https://www.youtube.com/live/ByED80IKdIU"},
    {"id": "cam02", "name": "Peace Bridge",       "yt_link": "https://www.youtube.com/live/DnUFAShZKus"},
]
CAM_MAP = {c['id']: c for c in VIDEO_LIST}
CAP_CACHE = {}

# Standard y-axis domains for metrics
DOMAINS = {
    'blur': (0, 2000),
    'brightness': (0, 255),
    'contrast': (0, 128),
    'noise': (0, 128),
    'delta': (0, 1.0)
}

# --------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------

def get_capture(mode, yt_link=None, upload=None):
    key = yt_link if mode != 'Local File' else upload.name
    if key in CAP_CACHE:
        return CAP_CACHE[key]
    if mode in ('Predefined', 'Custom URL'):
        proc = subprocess.run(
            ["yt-dlp", "-f", "best", "-g", yt_link],
            capture_output=True, text=True, check=True
        )
        stream_url = proc.stdout.splitlines()[0]
        cap = cv2.VideoCapture(stream_url)
    else:
        tmp_path = SCRIPT_DIR / upload.name
        if not tmp_path.exists():
            with open(tmp_path, 'wb') as f:
                f.write(upload.read())
        cap = cv2.VideoCapture(str(tmp_path))
    CAP_CACHE[key] = cap
    return cap


def append_unique(csv_path, row_dict):
    df_row = pd.DataFrame([row_dict])
    if csv_path.exists():
        df_exist = pd.read_csv(csv_path, parse_dates=['timestamp'])
        df_row = df_row[~df_row['timestamp'].isin(df_exist['timestamp'])]
    df_row.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)

# --------------------------------------------------------------------------------
# UI: Source Selection
# --------------------------------------------------------------------------------

st.title("📊 Traffic Data Quality Dashboard")
source_mode = st.radio("Source Type", ['Predefined', 'Custom URL', 'Local File'], horizontal=True)

if source_mode == 'Predefined':
    sel = st.selectbox("Select Camera", list(CAM_MAP.keys()), format_func=lambda k: CAM_MAP[k]['name'])
    yt_link = CAM_MAP[sel]['yt_link']
    upload = None
    log_id = sel
elif source_mode == 'Custom URL':
    yt_link = st.text_input("YouTube Live URL")
    upload = None
    log_id = urllib.parse.quote_plus(yt_link) if yt_link else None
else:
    upload = st.file_uploader("Upload Video File", type=['mp4','mov','avi'])
    yt_link = None
    log_id = upload.name if upload else None

if (source_mode != 'Local File' and not yt_link) or (source_mode == 'Local File' and not upload):
    st.warning("Please select or provide a data source.")
    st.stop()

# CSV path
csv_file = DATA_DIR / f"{log_id}_stats.csv"

# --------------------------------------------------------------------------------
# Real-Time Monitoring
# --------------------------------------------------------------------------------

enable_rt = (source_mode != 'Local File') and st.checkbox("Enable Real-Time Monitoring", value=False)
if enable_rt:
    cap = get_capture(source_mode, yt_link=yt_link, upload=upload)
    if not cap.isOpened():
        st.error("Cannot open video stream.")
        st.stop()

    # init CSV
    if not csv_file.exists():
        pd.DataFrame(columns=['timestamp','blur','brightness','contrast','noise','delta']).to_csv(csv_file, index=False)

    # placeholders
    cols = st.columns(4)
    ph_metrics = [c.empty() for c in cols]
    chart_cols = st.columns(4)
    ph_charts = [c.empty() for c in chart_cols]

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    exp_int = 1.0/fps if fps>0 else 0.03
    prev_ts = None
    total=0; stutters=0

    st.write(f"Streaming from: {log_id} (~{fps:.1f} FPS)")
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Stream ended or error.")
            break
        now = datetime.now()
        delta = (now-prev_ts).total_seconds() if prev_ts else exp_int
        prev_ts = now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        metrics = {
            'blur': cv2.Laplacian(gray, cv2.CV_64F).var(),
            'brightness': float(gray.mean()),
            'contrast': float(gray.std()),
            'noise': float(np.std(gray - cv2.GaussianBlur(gray,(5,5),0))),
            'delta': delta
        }
        append_unique(csv_file, {'timestamp':now, **metrics})

        total+=1
        if delta>exp_int*1.5: stutters+=1

        # update metric cards
        for i, key in enumerate(['blur','brightness','contrast','noise']):
            ph_metrics[i].metric(key.title(), f"{metrics[key]:.1f}")

        # update charts with fixed y-domain
        df_recent = pd.read_csv(csv_file, parse_dates=['timestamp']).set_index('timestamp')
        for i, key in enumerate(['blur','brightness','contrast','noise']):
            chart = alt.Chart(df_recent.reset_index()).mark_line().encode(
                x='timestamp:T',
                y=alt.Y(f'{key}:Q', scale=alt.Scale(domain=DOMAINS[key]))
            ).properties(title=key.title())
            ph_charts[i].altair_chart(chart, use_container_width=True)

        time.sleep(1)
    cap.release()

# --------------------------------------------------------------------------------
# Static Analysis
# --------------------------------------------------------------------------------
else:
    if source_mode == 'Local File':
        with st.spinner('Processing local video...'):
            cap = get_capture(source_mode, upload=upload)
            data=[]; frames=0
            while True:
                ret,frame=cap.read()
                if not ret: break
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                data.append({
                    'timestamp':frames,
                    'blur':cv2.Laplacian(gray,cv2.CV_64F).var(),
                    'brightness':float(gray.mean()),
                    'contrast':float(gray.std()),
                    'noise':float(np.std(gray-cv2.GaussianBlur(gray,(5,5),0)))
                })
                frames+=1
            cap.release()
        st.success(f'Processed {frames} frames.')
        df_local=pd.DataFrame(data).set_index('timestamp')
        for key in ['blur','brightness','contrast','noise']:
            chart=alt.Chart(df_local.reset_index()).mark_line().encode(
                x='timestamp:Q',
                y=alt.Y(f'{key}:Q', scale=alt.Scale(domain=DOMAINS[key]))
            ).properties(title=key.title())
            st.altair_chart(chart, use_container_width=True)
    else:
        if not csv_file.exists():
            st.info('No data. Enable real-time to collect.')
            st.stop()
        df_hist=pd.read_csv(csv_file, parse_dates=['timestamp']).set_index('timestamp')
        st.subheader('Historical Metrics')
        if 'delta' in df_hist:
            est_fps=1.0/df_hist['delta'].mean()
            st.metric('Estimated FPS',f'{est_fps:.2f}')
        for key in ['blur','brightness','contrast','noise','delta']:
            if key in df_hist:
                chart=alt.Chart(df_hist.reset_index()).mark_line().encode(
                    x='timestamp:T',
                    y=alt.Y(f'{key}:Q', scale=alt.Scale(domain=DOMAINS.get(key, None)))
                ).properties(title=key.title())
                st.altair_chart(chart, use_container_width=True)
