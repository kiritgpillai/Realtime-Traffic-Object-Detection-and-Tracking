import gradio as gr
import requests, os, tempfile

API_URL = "http://fastapi_server:8000/process"
DATA_DIR = "/app/data"

def inference_file(video_path):
    files = {"file": open(video_path, "rb")}
    r = requests.post(API_URL, files=files)
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=DATA_DIR)
    out.write(r.content)
    return out.name

with gr.Blocks() as demo:
    gr.Markdown("## 视频检测+跟踪 Demo")
    vid_in  = gr.Video(label="上传视频")
    btn     = gr.Button("Run Demo")
    vid_out = gr.Video(label="结果")
    btn.click(inference_file, inputs=vid_in, outputs=vid_out)

demo.launch(server_name="0.0.0.0", port=7860)


""" 

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

app = FastAPI()

@app.get("/", response_class=PlainTextResponse)
async def read_root():
    return "Hello, World!"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",      # 文件名:FastAPI 实例名
        host="0.0.0.0", # 让外部可访问
        port=5000,      # 你指定的 5000 端口
        reload=True     # 开发时自动热重载
    )
 """