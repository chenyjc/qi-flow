#!/bin/bash

set -e

echo "=========================================="
echo "  QiFlow 部署脚本"
echo "=========================================="

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRONTEND_DIR="$PROJECT_DIR/frontend"
BACKEND_DIR="$PROJECT_DIR/backend"
STATIC_DIR="$BACKEND_DIR/static"

echo "[1/5] 检查环境..."

if ! command -v node &> /dev/null; then
    echo "错误: 未找到 Node.js，请先安装 Node.js"
    exit 1
fi

# 检查 Python 命令
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "错误: 未找到 Python，请先安装 Python"
    exit 1
fi

echo "  Node.js: $(node -v)"
echo "  Python: $($PYTHON_CMD --version)"

echo ""
echo "[2/5] 安装前端依赖..."
cd "$FRONTEND_DIR"
npm install

echo ""
echo "[3/5] 构建前端..."
npm run build

echo ""
echo "[4/5] 部署静态文件到后端..."
rm -rf "$STATIC_DIR"
mkdir -p "$STATIC_DIR"
cp -r "$FRONTEND_DIR/dist/"* "$STATIC_DIR/"
echo "  静态文件已复制到: $STATIC_DIR"

echo ""
echo "[5/5] 安装后端依赖..."
cd "$BACKEND_DIR"
if [ -d ".venv" ]; then
    echo "  检测到虚拟环境，激活中..."
    source .venv/bin/activate
else
    echo "  提示: 未检测到虚拟环境，建议创建: $PYTHON_CMD -m venv .venv && source .venv/bin/activate"
fi
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "  部署完成!"
echo "=========================================="
echo ""
echo "启动服务:"
echo "  cd $BACKEND_DIR"
echo "  uvicorn backend.main:app --host 0.0.0.0 --port 8008"
echo ""
echo "访问地址:"
echo "  前端页面: http://<server-ip>:8008/"
echo "  API文档:  http://<server-ip>:8008/docs"
echo ""