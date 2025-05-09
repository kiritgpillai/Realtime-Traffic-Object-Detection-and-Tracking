from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
from ultralytics import YOLO

app = FastAPI()
MODEL_PATH = "/app/model.pt"        # 镜像里或者 block volume 挂载的路径
STREAM_URL = "https://www.youtube.com/live/BN7gzH-i-zo?feature=shared"

def frame_generator():
    # 1) 用 yt-dlp / OpenCV 拿到真实流地址（如果你已硬编码或在 ENV 里，就直接 cap = ...）
    cap = cv2.VideoCapture(STREAM_URL)
    model = YOLO(MODEL_PATH)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 2) YOLO 推理 + 画框
        results = model(frame)[0]
        for box in results.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        # 3) 编码 JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        chunk = jpeg.tobytes()
        # 4) 按 MJPEG 格式拼接
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + chunk + b'\r\n')

@app.get("/stream")
def stream_endpoint():
    return StreamingResponse(
        frame_generator(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )
