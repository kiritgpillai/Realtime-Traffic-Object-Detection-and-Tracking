import gradio as gr
import requests
import os, tempfile

API_URL = "http://fastapi_server:8000/process"  # FastAPI 推理端点

def inference_file(video):
    # video: 本地 temp 文件路径
    files = {"file": open(video.name, "rb")}
    r = requests.post(API_URL, files=files)
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out.write(r.content)
    return out.name

def inference_url(url):
    r = requests.post(API_URL, json={"stream_url": url})
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out.write(r.content)
    return out.name

with gr.Blocks() as demo:
    gr.Markdown("## 视频检测+跟踪 Demo")
    with gr.Tabs():
        with gr.TabItem("上传视频"):
            vid_in  = gr.Video(label="选择本地视频")
            btn1    = gr.Button("Run on File")
            vid_out = gr.Video(label="处理后视频")
            btn1.click(inference_file, inputs=vid_in, outputs=vid_out)
        with gr.TabItem("网络流地址"):
            url_in  = gr.Textbox(label="Stream URL")
            btn2    = gr.Button("Run on Stream")
            vid2_out= gr.Video(label="处理后视频")
            btn2.click(inference_url, inputs=url_in, outputs=vid2_out)
    demo.launch(server_name="0.0.0.0", port=7860)
