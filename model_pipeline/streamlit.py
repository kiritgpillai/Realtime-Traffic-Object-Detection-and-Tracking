import streamlit as st
import pandas as pd
import os
import altair as alt

# Path to directory containing stats CSVs
data_dir = "video_data"

# Utility: load and preprocess data for a given camera ID
def load_data(cam_id: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{cam_id}_stats.csv")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["minute"] = df["timestamp"].dt.floor("min")
    return df

# Main dashboard
def main():
    st.set_page_config(page_title="Traffic Data Quality Dashboard", layout="wide")
    st.title("Traffic Data Quality Dashboard")

    # List available camera IDs
    csv_files = [f for f in os.listdir(data_dir) if f.endswith("_stats.csv")]
    cam_ids = [f.replace("_stats.csv", "") for f in csv_files]

    # Interactive: camera selection and data reload
    selected_cam = st.selectbox("Select Camera Source", cam_ids)
    if st.button("Reload Data"):
        st.experimental_rerun()

    # Load data
    df = load_data(selected_cam)
    st.subheader(f"Data Source: {selected_cam}")

    # Interactive: time range slider
    min_time = df["minute"].min()
    max_time = df["minute"].max()
    start_time, end_time = st.slider(
        "Select Time Range", min_value=min_time, max_value=max_time,
        value=(min_time, max_time), format="YYYY-MM-DD HH:mm"
    )
    mask = (df["minute"] >= start_time) & (df["minute"] <= end_time)
    df_filtered = df.loc[mask]

    st.write(f"Filtered frames: {len(df_filtered)}")
    st.write(f"Average blur score: {df_filtered['blur_score'].mean():.2f}")

    # Blur score over time
    st.markdown("### Blur Score Over Time")
    line_chart = alt.Chart(df_filtered).mark_line().encode(
        x='minute:T',
        y='blur_score:Q'
    ).properties(width=800, height=300)
    st.altair_chart(line_chart, use_container_width=True)

    # Blur score distribution histogram
    st.markdown("### Blur Score Distribution")
    hist_chart = alt.Chart(df_filtered).mark_bar().encode(
        alt.X("blur_score:Q", bin=alt.Bin(maxbins=50)),
        y='count()'
    ).properties(width=400, height=300)
    st.altair_chart(hist_chart, use_container_width=True)

if __name__ == "__main__":
    main()
