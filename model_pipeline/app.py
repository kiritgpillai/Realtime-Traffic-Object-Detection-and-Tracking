

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
