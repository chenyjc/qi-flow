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