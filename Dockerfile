# 基于GitHub Actions构建的image
FROM ghcr.io/chenyjc/qi-flow:latest

# 暴露Streamlit默认端口（必须与原image一致）
EXPOSE 8501

# 启动命令（与原image一致）
ENTRYPOINT ["streamlit", "run", "streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]