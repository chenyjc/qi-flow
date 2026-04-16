# QiFlow 量化交易策略系统

基于 Qlib 和 FastAPI 的量化交易策略回测平台，提供完整的量化投资工作流。

## 名称含义

- **Q** - 量化 / Qlib
- **i** - 投资
- **Flow** - Pipeline 工作流

## 功能特性

- **数据管理** - 支持沪深300、中证500、中证800、中证1000等多市场数据下载与预览
- **模型训练** - 基于 Alpha158 因子的 GBDT 模型训练，支持 LightGBM、XGBoost
- **策略回测** - TopkDropoutStrategy 策略回测，支持自定义参数配置
- **结果分析** - 可视化展示收益曲线、持仓明细、关键绩效指标
- **实时进度** - SSE 流式返回训练和下载进度

## 技术栈

### 后端

| 技术 | 说明 |
|------|------|
| FastAPI | 现代 Python Web 框架 |
| Qlib | 微软量化投资平台 |
| MLflow | 实验跟踪和模型管理 |
| Akshare | 中国股票数据获取 |
| SQLite | 本地数据库存储 |

### 前端

| 技术 | 版本 |
|------|------|
| Vue 3 | ^3.4.0 |
| Element Plus | ^2.5.0 |
| Chart.js | ^4.4.0 |
| Vite | ^5.0.0 |

## 项目结构

```
qi-flow/
├── backend/                # FastAPI 后端
│   ├── routes/             # API 路由
│   ├── services/           # 业务服务
│   ├── static/             # 前端静态文件
│   └── main.py             # 应用入口
├── frontend/               # Vue 3 前端
│   ├── src/components/     # Vue 组件
│   └ vite.config.js        # Vite 配置
├── stmain/                 # Streamlit 应用 (Deprecated)
├── notebooks/              # Jupyter 测试笔记本
├── scripts/                # 运行脚本
├── deploy.sh               # 部署脚本
├── start_backend.sh        # 后端启动脚本
├── start_nohup.sh          # 后台启动脚本
├── stop_nohup.sh           # 停止脚本
└── qiflow.service          # systemd 服务配置
```

## 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+

### 部署步骤

```bash
# 1. 克隆项目
git clone https://github.com/your-repo/qi-flow.git
cd qi-flow

# 2. 执行部署脚本
chmod +x deploy.sh
./deploy.sh

# 3. 启动服务
./start_nohup.sh

# 4. 访问应用
# 前端页面: http://localhost:8008
# API文档:  http://localhost:8008/docs
```

### 手动部署

```bash
# 前端构建
cd frontend
npm install
npm run build

# 复制静态文件
cp -r frontend/dist/* backend/static/

# 后端启动
cd backend
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8008
```

## 训练方法详解

模型训练核心实现在 [`backend/services/qlib_service.py`](backend/services/qlib_service.py)，采用 Qlib 量化投资框架进行端到端的机器学习流水线训练。

### 1. 数据准备

使用 **Alpha158** 数据处理器生成158个量化因子特征：

```python
data_handler_config = {
    "start_time": train_start_date,    # 数据起始日期
    "end_time": test_end_date,          # 数据结束日期
    "fit_start_time": train_start_date, # 特征拟合起始日
    "fit_end_time": valid_end_date,     # 特征拟合结束日
    "instruments": market,              # 市场选择：csi300/csi500/csi800/csi1000
}
```

### 2. 模型配置

支持三种模型类型：**LightGBM**（默认）、**XGBoost**、**Linear**

**LightGBM 默认参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `learning_rate` | 0.0421 | 学习率 |
| `max_depth` | 8 | 树的最大深度 |
| `num_leaves` | 210 | 叶子节点数 |
| `subsample` | 0.8789 | 样本采样比例 |
| `colsample_bytree` | 0.8879 | 特征采样比例 |
| `lambda_l1` | 205.6999 | L1正则化系数 |
| `lambda_l2` | 580.9768 | L2正则化系数 |

```python
"model": {
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {
        "loss": "mse",          # 损失函数：均方误差
        "learning_rate": lr,
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "num_threads": 20,
        "random_state": 42,
    }
}
```

### 3. 数据集划分

采用三段式时间序列划分：

```python
"segments": {
    "train": (train_start_date, train_end_date),   # 训练集
    "valid": (valid_start_date, valid_end_date),   # 验证集
    "test": (test_start_date, test_end_date),      # 测试集
}
```

**默认时间范围**（动态计算）：
- **训练集**：2008-01-01 至 1年前
- **验证集**：1年前 至 3个月前
- **测试集**：3个月前 至 今天

### 4. 训练流程

```python
with R.start(experiment_name="train_model", recorder_name=train_recorder_name):
    # 1. 记录训练参数到 MLflow
    R.log_params(market=market, benchmark=benchmark, ...)
    
    # 2. 获取记录器 ID
    recorder = R.get_recorder()
    rid = recorder.id
    
    # 3. 执行模型训练
    model.fit(dataset)
    
    # 4. 保存训练好的模型
    R.save_objects(trained_model=model)
```

**关键特性**：
- 使用 **MLflow** 进行实验跟踪（SQLite 后端存储）
- 自动生成记录名称：`{时间}_{市场}_{训练起止日期}`
- 支持流式进度返回（SSE）

### 5. 策略回测

训练完成后使用 **TopkDropoutStrategy** 进行回测验证：

```python
port_analysis_config = {
    "strategy": {
        "class": "TopkDropoutStrategy",
        "kwargs": {
            "model": model,
            "dataset": dataset,
            "topk": 10,       # 持仓数量（Top10股票）
            "n_drop": 1,      # 每次调仓卖出数量
        }
    },
    "backtest": {
        "account": 1000000,               # 初始资金（100万）
        "benchmark": "SH000300",          # 基准指数
        "exchange_kwargs": {
            "freq": "day",                # 日线级别
            "limit_threshold": 0.095,     # 涨跌停限制
            "deal_price": "close",        # 成交价格：收盘价
            "open_cost": 0.0005,          # 开仓成本（万5）
            "close_cost": 0.0015,         # 平仓成本（万15）
            "min_cost": 5,                # 最低交易成本
        }
    }
}
```

### 6. 完整数据流

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ GitHub数据源 │ → │ Alpha158    │ → │ LightGBM    │ → │ TopkDropout │ → │ 回测结果    │
│ (日K线数据) │    │ 特征工程    │    │ 模型训练    │    │ 策略执行    │    │ (收益率/持仓)│
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

这是一个典型的 **量化投资机器学习流水线**：使用历史数据训练收益率预测模型，然后通过 Topk 选股策略模拟交易验证效果。

## API 接口

### Stock API (`/api/stock`)

| 接口 | 方法 | 功能 |
|------|------|------|
| `/csi300` | GET | 获取沪深300成分股 |
| `/history` | POST | 获取股票历史行情 |
| `/factors` | POST | 获取股票因子数据 |
| `/backtest` | POST | 简单策略回测 |

### Qlib API (`/api/qlib`)

| 接口 | 方法 | 功能 |
|------|------|------|
| `/download_data_stream` | GET (SSE) | 流式下载Qlib数据 |
| `/train_stream` | POST (SSE) | 流式训练模型 |
| `/backtest` | POST | 执行策略回测 |
| `/recorders` | GET | 获取训练记录 |
| `/backtest_recorders` | GET | 获取回测记录 |
| `/preview_data` | POST | 数据预览 |

## 支持的市场

| 市场代码 | 说明 |
|----------|------|
| csi300 | 沪深300 |
| csi500 | 中证500 |
| csi800 | 中证800 |
| csi1000 | 中证1000 |

## Systemd 服务（可选）

```bash
# 安装服务（修改 qiflow.service 中的路径）
sudo cp qiflow.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable qiflow
sudo systemctl start qiflow

# 管理命令
sudo systemctl status qiflow
sudo systemctl restart qiflow
sudo systemctl stop qiflow
```

## Docker 部署

```bash
docker build -t qiflow .
docker run -d -p 8008:8008 qiflow
```

## 许可证

Apache-2.0