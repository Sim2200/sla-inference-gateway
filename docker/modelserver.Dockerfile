FROM python:3.12-slim
RUN pip install --no-cache-dir onnxruntime==1.30.0 numpy pillow fastapi==0.141.1 "uvicorn[standard]==0.53.0" prometheus-client==0.26.0
WORKDIR /app
COPY src/modelserver ./modelserver
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "modelserver.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
