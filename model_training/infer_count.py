import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import subprocess
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# 1. Configuration
YOUTUBE_URL = "https://youtu.be/6dp-bvQ7RWo"
MODEL_PATH  = "model_pipeline/best.pt"

# 2. Get stream URL
proc = subprocess.run(
    ["yt-dlp", "-f", "best", "-g", YOUTUBE_URL],
    capture_output=True, text=True, check=True
)
stream_url = proc.stdout.splitlines()[0]

# 3. Open stream
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    raise RuntimeError("无法打开视频流")

# 4. Load model & tracker
model   = YOLO(MODEL_PATH)
tracker = DeepSort(max_age=30)

seen_ids = set()
CONF_THRESHOLD = 0.45  # 置信度阈值

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- 1) YOLO 检测 + 阈值过滤 ---
    results = model(frame)[0]
    dets = []
    for box in results.boxes:
        conf = float(box.conf[0])
        cls  = int(box.cls[0])
        # 只保留 'car' 且 conf >= 阈值
        if model.names[cls] != "car" or conf < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        dets.append(([x1, y1, x2-x1, y2-y1], conf, cls))

    # --- 2) DeepSORT 跟踪 ---
    tracks = tracker.update_tracks(dets, frame=frame)

    # --- 3) 累计唯一车辆数 & 画框+ID ---
    for tr in tracks:
        if not tr.is_confirmed():
            continue
        tid = tr.track_id
        seen_ids.add(tid)
        x1, y1, x2, y2 = tr.to_tlbr()
        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        cv2.putText(frame, f"ID:{tid}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # --- 4) 显示实时与累计数 ---
    current_count = sum(1 for t in tracks if t.is_confirmed())
    cv2.putText(frame, f"In frame: {current_count}",      (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Unique total: {len(seen_ids)}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    # --- 5) 展示 & 退出控制 ---
    cv2.imshow("Cars >=45% Conf", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Total unique cars appeared (conf>={CONF_THRESHOLD}): {len(seen_ids)}")
