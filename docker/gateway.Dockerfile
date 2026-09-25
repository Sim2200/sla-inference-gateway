FROM python:3.12-slim
COPY requirements-gateway.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-gateway.txt
WORKDIR /app
COPY src/gateway ./gateway
COPY src/tracing ./tracing
COPY deploy/registry.yaml ./registry.yaml
ENV GATEWAY_CONFIG=/app/registry.yaml PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["uvicorn", "gateway.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
