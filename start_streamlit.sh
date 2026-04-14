#!/bin/bash

# 启动脚本 - Streamlit 版本

echo "===== Streamlit 量化交易策略系统启动脚本 ====="
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: Python 3 未安装"
    exit 1
fi

echo "1. 安装依赖..."
pip install -r stmain/requirements.txt

if [ $? -ne 0 ]; then
    echo "错误: 依赖安装失败"
    exit 1
fi

echo ""
echo "2. 启动 Streamlit 应用..."
echo ""
echo "访问地址: http://localhost:8501"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

# 设置环境变量（可选设置密码）
# export APP_PASSWORD="your_password_here"

# 启动Streamlit
streamlit run stmain/app.py --server.port 8501 --server.address 0.0.0.0