#!/bin/bash

# 后台启动脚本：使用 nohup 运行后端服务（前后端合一）

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$ROOT_DIR/backend_nohup.pid"
LOG="$ROOT_DIR/backend_nohup.log"

is_running() {
    local pid="$1"
    [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1
}

start_background_script() {
    local script="$1"
    local log="$2"
    local pidfile="$3"

    if command -v setsid >/dev/null 2>&1; then
        nohup setsid bash "$script" > "$log" 2>&1 &
    else
        nohup bash "$script" > "$log" 2>&1 &
    fi

    local pid=$!
    echo "$pid" > "$pidfile"
    echo "$pid"
}

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if is_running "$PID"; then
        echo "后台服务已在运行：pid=$PID"
        exit 0
    else
        rm -f "$PID_FILE"
    fi
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ -f "$LOG" ]; then
    mv "$LOG" "$ROOT_DIR/backend_nohup_$TIMESTAMP.log"
fi

echo "启动后台服务，日志写入：$LOG"
BACKEND_PID=$(start_background_script "$ROOT_DIR/start_backend.sh" "$LOG" "$PID_FILE")

echo "启动完成。"
echo "backend pid=$BACKEND_PID"
echo "访问地址: http://localhost:8008"