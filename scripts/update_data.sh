#!/bin/bash

# Qlib 数据下载脚本（带进度输出）
# 输出格式: PROGRESS:百分比:消息

set -e

QLIB_DIR="$HOME/.qlib"
DATA_DIR="$QLIB_DIR/qlib_data/cn_data"
BACKUP_DIR="$QLIB_DIR/qlib_data/cn_data_backup"
TEMP_FILE="$QLIB_DIR/qlib_bin.tar.gz"
URL="https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz"

echo "PROGRESS:0:开始下载Qlib数据..."

# 创建目录
mkdir -p "$QLIB_DIR"

echo "PROGRESS:5:检查文件大小..."

# 获取文件大小
# FILE_SIZE=$(curl -sI "$URL" | grep -i Content-Length | awk '{print $2}' | tr -d '\r\n' 2>/dev/null || echo "")
# if [ -n "$FILE_SIZE" ] && [ "$FILE_SIZE" -gt 0 ]; then
#     SIZE_MB=$((FILE_SIZE / 1024 / 1024))
#     echo "PROGRESS:10:文件大小: ${SIZE_MB}MB"
# fi

# echo "PROGRESS:15:开始下载..."

# # 下载文件（不显示进度条，避免非UTF-8字符）
# wget -q -O "$TEMP_FILE" "$URL" && {
#     echo "PROGRESS:80:下载完成"
# } || {
#     echo "PROGRESS:20:使用curl下载..."
#     curl -sL -o "$TEMP_FILE" "$URL"
#     echo "PROGRESS:80:下载完成"
# }

# 备份旧数据
if [ -d "$DATA_DIR" ]; then
    echo "PROGRESS:85:备份旧数据..."
    rm -rf "$BACKUP_DIR"
    mv "$DATA_DIR" "$BACKUP_DIR"
fi

echo "PROGRESS:88:开始解压..."

# 解压文件
mkdir -p "$DATA_DIR"
tar -xzf "$TEMP_FILE" -C "$DATA_DIR" --strip-components=1

echo "PROGRESS:95:解压完成"

# 删除临时文件
rm -f "$TEMP_FILE"

echo "PROGRESS:100:数据下载完成"

echo "SUCCESS:Qlib数据已更新完成"