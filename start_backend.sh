#!/bin/bash

# 启动脚本 - FastAPI + H5 前后端合一版本

echo "===== FastAPI + H5 量化交易策略系统启动脚本 ====="
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: Python 3 未安装"
    exit 1
fi

echo "1. 安装依赖..."
pip install -r backend/requirements.txt

if [ $? -ne 0 ]; then
    echo "错误: 依赖安装失败"
    exit 1
fi

echo ""
echo "2. 启动 FastAPI 服务（前后端合一）..."
echo ""
echo "访问地址: http://localhost:8000"
echo "API文档: http://localhost:8000/docs"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

# 启动FastAPI（前后端合一，静态文件由FastAPI提供）
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000 --log-level info