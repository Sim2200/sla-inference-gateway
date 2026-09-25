# SLA-Aware Inference Gateway

A gateway that keeps image-classification latency inside an SLA by routing each request between an
**accurate** model tier and a **fast** tier, based on the accurate tier's live p95 latency.

> **Status: in progress.** The model pipeline, gateway, model server, deploy configs and Core ML
> benchmark are done, and their results are below. The end-to-end load experiments (`make experiments`)
> are still being run; their results will be added here.

## How it works

```
client ──► gateway ──┬──► accurate tier  (ResNet-50 v2, int8 ONNX)
                     └──► fast tier      (MobileNetV3-Large, fp32 ONNX)
```

- **SLA controller** (`src/gateway/controller.py`): every tick it checks the accurate tier's p95 over
  the last window. Above the high-water mark (or with too many errors) it shifts traffic to the fast
  tier at once; it only gives traffic back after a cooldown below the low-water mark, so the split
  doesn't flap.
- **Admission control**: when the accurate tier already has its maximum requests in flight, a request
  spills to the fast tier immediately, which catches bursts shorter than one tick.
- **Circuit breakers** (`src/gateway/breaker.py`): closed / open / half-open per backend.
- **Canary rollouts** (`src/gateway/canary.py`): a new accurate-tier version goes 10% → 50% → 100%,
  compared against the stable version on error rate and p95 at each stage, with automatic rollback.
- **Baselines**: `always_accurate` and `always_fast` modes are plain proxies to one tier, used as the
  comparison in the benchmarks.
- **Ops**: Prometheus metrics and a Grafana dashboard (`deploy/`), Docker Compose for local runs, and
  a kind cluster with an HPA for the autoscaling experiment (`deploy/k8s/`).

## Choosing the tiers: ONNX model results

Top-1/top-5 on 9,500 held-out ImageNetV2 (matched-frequency) images, which the pretrained models never
saw. int8 = ONNX Runtime static quantization (QDQ, per-channel weights, calibrated on 500 separate
images). Latency is single-image ONNX Runtime inference with 2 CPU threads.

| Model | Precision | Size (MB) | Top-1 | Top-5 | p50 / p95 (ms) |
|---|---|---|---|---|---|
| MobileNetV3-Large | fp32 | 21.9 | 60.58% | 82.48% | 9.9 / 16.1 |
| MobileNetV3-Large | int8 | 6.0 | 56.61% | 79.13% | 9.7 / 13.2 |
| ResNet-18 | fp32 | 46.8 | 57.51% | 80.13% | 23.5 / 29.2 |
| ResNet-18 | int8 | 11.8 | 57.24% | 79.89% | 15.1 / 17.5 |
| ResNet-50 v1 | fp32 | 102.2 | 63.40% | 84.78% | 58.4 / 90.4 |
| ResNet-50 v1 | int8 | 26.1 | 63.11% | 84.53% | 38.6 / 56.2 |
| ResNet-50 v2 | fp32 | 102.2 | 70.08% | 88.85% | 61.8 / 132.7 |
| **ResNet-50 v2** | **int8** | **26.1** | **68.77%** | **88.37%** | **41.3 / 85.7** |

**Accurate tier: ResNet-50 v2 int8.** It keeps most of fp32's accuracy (−1.3 pt top-1) at about a
third less p50 latency and a quarter of the size. **Fast tier: MobileNetV3 fp32**, because int8 cost
it 4 pt of top-1 for no real p50 gain.

## On-device inference with Core ML

`models/coreml_bench.py` converts the same models to Core ML (fp32, fp16, and int8 linear per-channel
weight quantization) and measures them on a Mac. Top-1 is on 2,000 ImageNetV2 evaluation images;
latency is single-image predict over 200 runs after warm-up. Hardware: Intel i7-9750H MacBook Pro
with an AMD Radeon Pro 5300M (no Neural Engine).

| Model | Precision | Size (MB) | Top-1 | CPU p50 / p95 (ms) | CPU+GPU p50 / p95 (ms) |
|---|---|---|---|---|---|
| ResNet-50 v2 | fp32 | 102.2 | 70.05% | 85.6 / 89.1 | 11.5 / 12.3 |
| ResNet-50 v2 | fp16 | 51.2 | 70.05% | 91.1 / 95.6 | 11.8 / 12.8 |
| ResNet-50 v2 | int8 | 25.7 | 70.40% | 88.0 / 90.6 | 12.1 / 12.8 |
| MobileNetV3-Large | fp32 | 22.0 | 59.75% | 12.0 / 12.4 | 6.7 / 8.3 |
| MobileNetV3-Large | fp16 | 11.1 | 59.75% | 14.3 / 15.7 | 6.2 / 6.7 |
| MobileNetV3-Large | int8 | 5.7 | 60.15% | 14.1 / 15.7 | 6.1 / 6.6 |

- int8 weights make ResNet-50 **4x smaller with no top-1 loss** here.
- Running on the GPU is **about 7x faster** than CPU for ResNet-50.
- int8 doesn't speed up the CPU path: Core ML's weight-only quantization stores int8 weights and
  dequantizes them to compute, so it saves size and memory rather than compute. ONNX Runtime's static
  quantization above also quantizes activations, which is why it does cut CPU latency.

## Running it

```bash
make data          # ImageNetV2 matched-frequency (1.2 GB)
make models        # export ONNX candidates + int8 variants
make evaluate      # accuracy + latency table above
make test          # gateway unit + integration tests (no Docker)
make up            # gateway :8080, Prometheus :9090, Grafana :3000
make experiments   # load experiments (about 1.5 hours)
make coreml        # Core ML conversion + on-device benchmark (macOS)
make k8s-up        # kind cluster with HPA; then make k8s-hpa
```

Results are written to `results/`.
