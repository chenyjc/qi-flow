FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    gcc \
    g++ \
    make \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements.txt
COPY requirements.txt .

# 安装Python依赖，增加超时和重试机制
RUN pip install --no-cache-dir --timeout=120 --retries=5 -r requirements.txt

# 复制应用文件
COPY . .

# 暴露Streamlit默认端口
EXPOSE 8501

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 设置Streamlit环境变量
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_ENABLE_CORS=true
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# 运行Streamlit应用
CMD ["streamlit", "run", "streamlit/app.py"]