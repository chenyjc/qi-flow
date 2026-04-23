"""
QiFlow 量化交易策略系统

基于 Qlib 和 FastAPI 的量化交易策略回测系统

API 路由:
=========
/api/qlib/*  - Qlib相关接口 (主要接口)
/api/stock/* - 股票数据接口 (Deprecated)

文档:
  - API文档: /docs (Swagger UI)
  - 备用文档: /redoc (ReDoc)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .routes import stock, qlib, dl_models
import os
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="QiFlow量化交易策略系统",
    description="基于Qlib和FastAPI的量化交易策略回测系统",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
# stock 路由已被弃用，建议迁移到 qlib 路由
app.include_router(stock.router, prefix="/api/stock", tags=["stock (Deprecated)"])
app.include_router(qlib.router, prefix="/api/qlib", tags=["qlib"])
app.include_router(dl_models.router, prefix="/api/dl", tags=["deep_learning"])

# 配置静态文件服务
static_dir = os.path.join(os.path.dirname(__file__), "static")
assets_dir = os.path.join(static_dir, "assets")
if os.path.exists(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

@app.get("/")
async def root():
    """返回前端页面"""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "QiFlow量化交易策略系统"}

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008, reload=False)
