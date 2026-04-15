#!/bin/bash

# 后台停止脚本：停止 start_nohup.sh 启动的后台服务

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_PID_FILE="$ROOT_DIR/backend_nohup.pid"
FRONTEND_PID_FILE="$ROOT_DIR/frontend_nohup.pid"

function stop_pid_file() {
    local pid_file="$1"
    local name="$2"

    if [ ! -f "$pid_file" ]; then
        echo "$name PID 文件不存在：$pid_file"
        return
    fi

    local pid
    pid=$(cat "$pid_file")
    if [ -z "$pid" ]; then
        echo "$name PID 为空，删除 PID 文件"
        rm -f "$pid_file"
        return
    fi

    if kill -0 "$pid" >/dev/null 2>&1; then
        echo "停止 $name (pid=$pid) ..."
        kill "$pid" >/dev/null 2>&1
        sleep 1
        if kill -0 "$pid" >/dev/null 2>&1; then
            echo "$name 仍在运行，强制终止..."
            kill -9 "$pid" >/dev/null 2>&1
        fi
    else
        echo "$name pid=$pid 未在运行"
    fi

    rm -f "$pid_file"
}

stop_pid_file "$BACKEND_PID_FILE" "backend"
stop_pid_file "$FRONTEND_PID_FILE" "frontend"

echo "停止完成。"
