from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Summary
import tempfile
import shutil
import os
import time

from inference_yolo import download_model_from_minio, run_yolo_inference, upload_image_to_minio

app = FastAPI()

inference_time = Summary("inference_duration_seconds", "Time spent on YOLO inference")

# 在应用启动时预加载模型
MODEL_PATH = download_model_from_minio()

Instrumentator().instrument(app).expose(app)

# 添加健康检查端点
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_path": MODEL_PATH}

# 添加根路径处理器
@app.get("/")
def read_root():
    return {
        "service": "YOLO Image Inference API",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Detect objects in uploaded image",
            "/health": "GET - Check service health",
            "/metrics": "GET - Prometheus metrics"
        }
    }

# 将predict改为非异步函数
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    tmp_path = None
    try:
        # 保存上传的文件到临时位置
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        file.file.close()  # 关闭文件流
        
        # 进行推理
        start = time.time()
        result, image_with_boxes_path = run_yolo_inference(MODEL_PATH, tmp_path)
        end = time.time()
        
        # 记录推理时间
        inference_duration = end - start
        inference_time.observe(inference_duration)
        
        # 添加推理时间到结果
        result["inference_time_sec"] = round(inference_duration, 4)
        
        # 上传带有边界框的图像到MinIO
        minio_path = upload_image_to_minio(image_with_boxes_path)
        result["result_image"] = minio_path
        
        # 返回结果
        return result  # FastAPI会自动将字典转换为JSON响应

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"推理错误: {str(e)}\n{error_details}")
        return JSONResponse(
            status_code=500, 
            content={"error": str(e), "type": type(e).__name__}
        )

    finally:
        # 清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                print(f"无法删除临时文件 {tmp_path}: {str(e)}")