#!/bin/bash

# 后台停止脚本：停止 start_nohup.sh 启动的后台服务

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$ROOT_DIR/backend_nohup.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "PID 文件不存在：$PID_FILE"
    exit 0
fi

PID=$(cat "$PID_FILE")
if [ -z "$PID" ]; then
    echo "PID 为空，删除 PID 文件"
    rm -f "$PID_FILE"
    exit 0
fi

if kill -0 "$PID" >/dev/null 2>&1; then
    echo "停止后台服务 (pid=$PID) ..."
    kill -TERM "-$PID" >/dev/null 2>&1 || true
    sleep 1
    if kill -0 "$PID" >/dev/null 2>&1; then
        echo "服务仍在运行，强制终止..."
        kill -9 "-$PID" >/dev/null 2>&1 || true
    fi
else
    echo "pid=$PID 未在运行"
fi

rm -f "$PID_FILE"
echo "停止完成。"