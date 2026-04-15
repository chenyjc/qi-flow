#!/bin/bash

# 后台启动脚本：使用 nohup 运行 start_backend.sh 和 start_frontend.sh

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_PID_FILE="$ROOT_DIR/backend_nohup.pid"
FRONTEND_PID_FILE="$ROOT_DIR/frontend_nohup.pid"
BACKEND_LOG="$ROOT_DIR/backend_nohup.log"
FRONTEND_LOG="$ROOT_DIR/frontend_nohup.log"

function is_running() {
    local pid="$1"
    [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1
}

if [ -f "$BACKEND_PID_FILE" ]; then
    PID=$(cat "$BACKEND_PID_FILE")
    if is_running "$PID"; then
        echo "后台服务已在运行：backend pid=$PID"
        exit 0
    else
        rm -f "$BACKEND_PID_FILE"
    fi
fi

if [ -f "$FRONTEND_PID_FILE" ]; then
    PID=$(cat "$FRONTEND_PID_FILE")
    if is_running "$PID"; then
        echo "前端服务已在运行：frontend pid=$PID"
        exit 0
    else
        rm -f "$FRONTEND_PID_FILE"
    fi
fi

echo "启动后台服务，日志写入：$BACKEND_LOG"
nohup bash "$ROOT_DIR/start_backend.sh" > "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!
echo "$BACKEND_PID" > "$BACKEND_PID_FILE"

echo "启动前端服务，日志写入：$FRONTEND_LOG"
nohup bash "$ROOT_DIR/start_frontend.sh" > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
echo "$FRONTEND_PID" > "$FRONTEND_PID_FILE"

echo "启动完成。"
echo "backend pid=$BACKEND_PID"
echo "frontend pid=$FRONTEND_PID"
