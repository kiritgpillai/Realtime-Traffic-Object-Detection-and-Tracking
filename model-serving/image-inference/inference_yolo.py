import os
import tempfile
import datetime
from pathlib import Path

from minio import Minio
from ultralytics import YOLO


MINIO_ENDPOINT = "129.114.27.202:30000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "miniopassword"
MINIO_SECURE = False
BUCKET_NAME = "results"

MODEL_CACHE_PATH = os.path.expanduser("~/.cache/yolo_model/best.pt")
MODEL_REMOTE_PATH = "5/ccd4f18c17fa4979973f1de130bdd83a/artifacts/models/best.pt"


def download_model_from_minio() -> str:
    os.makedirs(os.path.dirname(MODEL_CACHE_PATH), exist_ok=True)

    if os.path.exists(MODEL_CACHE_PATH) and os.path.getsize(MODEL_CACHE_PATH) > 0:
        print(f"Cache: {MODEL_CACHE_PATH}")
        return MODEL_CACHE_PATH

    client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=MINIO_SECURE)

    if not client.bucket_exists("mlflow"):
        raise RuntimeError("Bucket 'mlflow' 不存在")

    tmp_model = tempfile.NamedTemporaryFile(delete=False)
    client.fget_object("mlflow", MODEL_REMOTE_PATH, tmp_model.name)
    tmp_model.close()

    os.replace(tmp_model.name, MODEL_CACHE_PATH)
    print(f"Download model: {MODEL_CACHE_PATH}")
    return MODEL_CACHE_PATH


def run_yolo_inference(model_path: str, image_path: str) -> tuple:
    model = YOLO(model_path)
    print(f"Load model: {model_path}")

    results = model.predict(
        source=image_path,
        conf=0.4,
        iou=0.45,
        max_det=300,
        save=True,
        save_txt=False,
        project=tempfile.gettempdir(),
        name="predict",
        exist_ok=True,
        verbose=True,
    )

    result = results[0]
    output = {
        "image": Path(image_path).name,
        "num_detections": len(result.boxes),
        "labels": {},
    }

    for box in result.boxes:
        cls = int(box.cls[0].item())
        label = model.names[cls]
        output["labels"][label] = output["labels"].get(label, 0) + 1

    saved_path = os.path.join(tempfile.gettempdir(), "predict", Path(image_path).name)
    return output, saved_path


def upload_image_to_minio(local_path: str) -> str:
    client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=MINIO_SECURE)

    if not client.bucket_exists(BUCKET_NAME):
        client.make_bucket(BUCKET_NAME)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    object_name = f"predict_results/predict_{timestamp}.jpg"
    client.fput_object(BUCKET_NAME, object_name, local_path, content_type="image/jpeg")

    print(f"Upload to MinIO: s3://{BUCKET_NAME}/{object_name}")
    return f"s3://{BUCKET_NAME}/{object_name}"
