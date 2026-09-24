PY      := .venv/bin/python
COMPOSE := docker compose
KIND    := kind
KUBECTL := kubectl
CLUSTER := sla

.PHONY: setup data models evaluate test up canary down experiments charts coreml \
        k8s-up k8s-hpa k8s-down clean

setup:                ## host venv for the gateway tests
	python3 -m venv .venv && .venv/bin/pip install -r requirements-gateway.txt pytest

data:                 ## download ImageNetV2 matched-frequency (1.2 GB, 10,000 images)
	mkdir -p data && curl -L https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz | tar xz -C data

models:               ## export all candidates to ONNX and build the int8 variants
	docker build -t slagw-builder -f docker/builder.Dockerfile docker
	docker run --rm -v "$(PWD)":/work slagw-builder python models/export.py

evaluate:             ## accuracy on 9,500 held-out images, then latency under a 2-CPU limit
	docker run --rm -v "$(PWD)":/work slagw-builder python models/evaluate.py accuracy --threads 10
	docker run --rm --cpus 2 -v "$(PWD)":/work slagw-builder python models/evaluate.py latency --threads 2

test:                 ## unit + integration tests for the gateway (no Docker needed)
	$(PY) -m pytest -q

up:                   ## gateway :8080, Prometheus :9090, Grafana :3000
	$(COMPOSE) up -d --build --wait gateway prometheus grafana

canary:               ## also start the servers for the canary experiment (v1 stable, faulty v2)
	$(COMPOSE) --profile canary up -d --wait accurate-v1 accurate-bad

down:
	$(COMPOSE) --profile canary --profile tools down

experiments:          ## all benchmark experiments (about 1.5 hours)
	$(COMPOSE) --profile tools build loadgen
	python3 loadtest/experiments.py all

coreml:               ## Core ML conversion (fp32/fp16/int8) + on-device benchmark (macOS only)
	uv venv -q -p 3.12 .venv-coreml
	uv pip install -q -p .venv-coreml/bin/python torch==2.2.2 torchvision==0.17.2 "numpy<2" coremltools pillow
	.venv-coreml/bin/python models/coreml_bench.py --eval-images 2000 --latency-runs 200

charts:               ## render results/*.json into report/figures
	$(COMPOSE) run --rm -T loadgen python loadtest/charts.py

k8s-up:               ## kind cluster with metrics-server, both tiers, HPA and the gateway
	$(KIND) create cluster --name $(CLUSTER) --config deploy/k8s/kind-config.yaml
	$(KIND) load docker-image --name $(CLUSTER) slagw-modelserver slagw-gateway slagw-loadgen
	$(KUBECTL) apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
	$(KUBECTL) -n kube-system patch deployment metrics-server --type=json \
	  -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
	$(KUBECTL) apply -f deploy/k8s/models.yaml -f deploy/k8s/hpa.yaml -f deploy/k8s/gateway.yaml
	$(KUBECTL) -n sla rollout status deploy/accurate deploy/fast deploy/gateway --timeout=180s

k8s-hpa:              ## autoscaling experiment on the kind cluster
	python3 loadtest/k8s_hpa.py

k8s-down:
	$(KIND) delete cluster --name $(CLUSTER)

clean:
	rm -rf results/raw .pytest_cache
