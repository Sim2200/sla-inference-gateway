# Offline model tooling: export to ONNX, quantize, evaluate. Not used at serving time.
FROM python:3.12-slim
RUN pip install --no-cache-dir \
        torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir onnx==1.19.0 onnxruntime==1.30.0 numpy pillow
ENV TORCH_HOME=/work/.cache/torch PYTHONUNBUFFERED=1
WORKDIR /work
