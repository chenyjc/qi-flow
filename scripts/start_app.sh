#!/bin/bash

# 添加定时任务
cat > /etc/cron.d/update-qlib-data << EOF
# 每天UTC 13:00执行完整流程：下载数据、训练、回测
0 13 * * * root /app/scripts/full_process.sh >> /var/log/cron.log 2>&1
EOF

# 给定时任务文件添加执行权限
chmod 0644 /etc/cron.d/update-qlib-data

# 给完整流程脚本添加执行权限
chmod +x /app/scripts/full_process.sh

# 启动cron服务
service cron start

# 执行完整流程脚本（首次启动时执行一次）
/app/scripts/full_process.sh

# 启动Streamlit应用
streamlit run streamlit/app.py