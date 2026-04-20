#!/bin/bash

# [Deprecated] 该脚本已不再维护
# 请使用 start_backend.sh 启动前后端合一的服务
#
# 原独立前端启动方式已被弃用，现在前后端已整合
# 新系统使用 FastAPI 提供静态文件服务
# 访问地址: http://localhost:8008

echo "===== 前端启动脚本 ====="
echo ""
echo "[警告] 该脚本已弃用，请使用 ./start_backend.sh 启动新系统"
echo ""
echo "新系统访问地址: http://localhost:8008"
echo ""

# 检查 Node.js 环境
if ! command -v node &> /dev/null; then
    echo "错误：Node.js 未安装"
    exit 1
fi

echo "Node.js 版本:"
node -v
echo ""

# 进入前端目录
cd "$(dirname "$0")/frontend"

# 检查 node_modules 是否存在
if [ ! -d "node_modules" ]; then
    echo "1. 安装前端依赖..."
    npm install
    
    if [ $? -ne 0 ]; then
        echo "错误：依赖安装失败"
        exit 1
    fi
fi

echo ""
echo "2. 启动前端开发服务器..."
echo ""
echo "访问地址：http://localhost:5173"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

# 启动 Vite 开发服务器
npm run dev
