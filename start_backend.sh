#!/bin/bash

# 启动脚本 - FastAPI + H5 前后端合一版本
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

echo "===== FastAPI + H5 量化交易策略系统启动脚本 ====="
echo ""

# 检查 Python 解释器
PYTHON_CMD=""
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD=python
else
    echo "错误: 找不到 Python 解释器"
    exit 1
fi

# 创建虚拟环境
if [ ! -d "$VENV_DIR" ]; then
    echo "检测到虚拟环境不存在，创建 .venv ..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "错误: 创建虚拟环境失败"
        exit 1
    fi
fi

# 激活虚拟环境
if [ -f "$VENV_DIR/bin/activate" ]; then
    # Unix / Linux / git bash
    source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    # Windows Git Bash / WSL
    source "$VENV_DIR/Scripts/activate"
else
    echo "错误: 找不到虚拟环境激活脚本"
    exit 1
fi

echo "虚拟环境已激活：$VENV_DIR"

echo "1. 安装依赖..."
pip install -r backend/requirements.txt

if [ $? -ne 0 ]; then
    echo "错误: 依赖安装失败"
    exit 1
fi

echo ""
echo "2. 启动 FastAPI 服务（前后端合一）..."
echo ""
echo "访问地址: http://localhost:8008"
echo "API文档: http://localhost:8008/docs"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

# 启动FastAPI（前后端合一，静态文件由FastAPI提供）
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8008 --log-level info