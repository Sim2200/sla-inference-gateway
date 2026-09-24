FROM python:3.12-slim
RUN pip install --no-cache-dir httpx==0.28.1 numpy matplotlib pyyaml==6.0.3
WORKDIR /work
ENV PYTHONUNBUFFERED=1
