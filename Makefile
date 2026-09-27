PY      := .venv/bin/python
COMPOSE := docker compose
KIND    := kind
KUBECTL := kubectl
CLUSTER := sla

.PHONY: setup data models evaluate test up canary down experiments charts coreml power \
        k8s-up k8s-hpa k8s-down gke-up gke-run gke-down clean

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

power:                ## energy per inference with powermetrics (macOS, needs sudo -v first)
	.venv-coreml/bin/python models/power_bench.py
	.venv-coreml/bin/python loadtest/charts.py

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

GCP_PROJECT ?= your-gcp-project
GCP_ZONE    ?= us-central1-a
AR          := us-central1-docker.pkg.dev/$(GCP_PROJECT)/slagw

gke-up:               ## GKE Standard cluster (e2-standard-4, node autoscaling 1-4) + images in Artifact Registry
	gcloud artifacts repositories create slagw --repository-format=docker --location=us-central1 --project $(GCP_PROJECT) || true
	gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
	docker build -t $(AR)/modelserver:v1 -f docker/modelserver.gke.Dockerfile .
	docker build -t $(AR)/loadgen:v1 -f docker/loadgen.gke.Dockerfile .
	docker tag slagw-gateway $(AR)/gateway:v1
	docker push $(AR)/modelserver:v1 && docker push $(AR)/loadgen:v1 && docker push $(AR)/gateway:v1
	gcloud container clusters create slagw --zone $(GCP_ZONE) --project $(GCP_PROJECT) \
	  --machine-type e2-standard-4 --num-nodes 2 --enable-autoscaling --min-nodes 1 --max-nodes 4
	gcloud container clusters get-credentials slagw --zone $(GCP_ZONE) --project $(GCP_PROJECT)
	sed 's#project-1adf2361-a5bc-4d4a-a7f#$(GCP_PROJECT)#g' deploy/gke/stack.yaml | $(KUBECTL) apply -f -
	$(KUBECTL) -n sla rollout status deploy/accurate deploy/fast deploy/gateway --timeout=300s

gke-run:              ## the spike experiment on GKE; writes results/gke.json and results/raw/gke_*
	python3 loadtest/gke_experiment.py

gke-down:             ## delete the cluster (the only thing that costs money while idle)
	gcloud container clusters delete slagw --zone $(GCP_ZONE) --project $(GCP_PROJECT) --quiet

clean:
	rm -rf results/raw .pytest_cache
