import subprocess
import cv2
import pandas as pd
import os
from datetime import datetime
from pathlib import Path

# Determine project root (two levels up from this script)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
# Directory to store stats CSVs at project root
SAVE_ROOT = PROJECT_ROOT / "video_data"
print(f"Saving stats to: {SAVE_ROOT}")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

# List of video sources (replace with actual links)
VIDEO_LIST = [
    {"id": "cam01", "name": "4 Corners Camera Downtown", "yt_link": "https://www.youtube.com/live/ByED80IKdIU?si=AhsjrctZolhPaX7o"},
    {"id": "cam02", "name": "Peace Bridge - Canada Bound", "yt_link": "https://www.youtube.com/live/DnUFAShZKus?si=NERZWgFOnwF1s1KX"},
    # add more sources as needed
]

# Get direct video stream URL using yt-dlp
# Prints debug info if fails
def get_stream_url(yt_url: str) -> str:
    try:
        result = subprocess.run(
            ["yt-dlp", "-f", "best", "-g", yt_url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        stream_url = result.stdout.strip().splitlines()[0]
        print(f"Obtained stream URL: {stream_url}")
        return stream_url
    except subprocess.CalledProcessError as e:
        print(f"Error obtaining stream URL: {e.stderr}")
        return ""

# Extract blur statistics by sequential reading
def extract_stats(video_id: str, stream_url: str, interval_sec: int = 3, max_samples: int = 100):
    if not stream_url:
        print(f"Skipping {video_id}: no stream URL")
        return
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"Error: Unable to open stream for {video_id}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval_frames = max(int(fps * interval_sec), 1)
    print(f"Stream FPS: {fps}, interval frames: {interval_frames}")

    stats = []
    frame_count = 0
    samples = 0
    while samples < max_samples:
        ret, frame = cap.read()
        if not ret or frame is None:
            frame_count += 1
            continue
        if frame_count % interval_frames == 0:
            blur_score = cv2.Laplacian(frame, cv2.CV_64F).var()
            timestamp = datetime.now().isoformat()
            stats.append({
                "frame_idx": frame_count,
                "timestamp": timestamp,
                "blur_score": blur_score,
            })
            samples += 1
        frame_count += 1
    cap.release()

    # Save to CSV in SAVE_ROOT
    df = pd.DataFrame(stats)
    csv_path = SAVE_ROOT / f"{video_id}_stats.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} samples for {video_id} to {csv_path}")

if __name__ == "__main__":
    for source in VIDEO_LIST:
        stream_url = get_stream_url(source["yt_link"])
        print(f"Processing {source['id']}...")
        extract_stats(source["id"], stream_url)
