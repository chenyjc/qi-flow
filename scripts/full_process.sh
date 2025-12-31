#!/bin/bash

# 执行完整流程：下载数据 -> 训练模型 -> 回测模型

echo "开始执行完整流程: $(date)"

# 1. 下载最新数据
echo "1. 开始下载最新数据..."
/app/scripts/update_data.sh

# 2. 执行训练模型
echo "2. 开始训练模型..."
python -c "
import sys
sys.path.append('/app')
from streamlit.app import train_model
train_model()
"

# 3. 执行回测模型
echo "3. 开始回测模型..."
python -c "
import sys
sys.path.append('/app')
from streamlit.app import backtest_model
backtest_model()
"

echo "完整流程执行完成: $(date)"
