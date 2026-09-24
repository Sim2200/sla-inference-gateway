# Core ML on-device benchmark

Hardware: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz, macOS 26.6.2. Top-1 on 2000 ImageNetV2 evaluation images; latency is single-image predict over 200 runs after warm-up.

| Model | Precision | Size (MB) | Top-1 | CPU p50 / p95 (ms) | CPU+GPU p50 / p95 (ms) |
|---|---|---|---|---|---|
| resnet50_v2 | fp32 | 102.2 | 70.05% | 85.59 / 89.1 | 11.48 / 12.28 |
| resnet50_v2 | fp16 | 51.15 | 70.05% | 91.05 / 95.6 | 11.83 / 12.76 |
| resnet50_v2 | int8 | 25.72 | 70.40% | 87.99 / 90.56 | 12.11 / 12.8 |
| mobilenet_v3_large | fp32 | 22.0 | 59.75% | 11.95 / 12.41 | 6.69 / 8.33 |
| mobilenet_v3_large | fp16 | 11.07 | 59.75% | 14.28 / 15.71 | 6.16 / 6.7 |
| mobilenet_v3_large | int8 | 5.7 | 60.15% | 14.14 / 15.67 | 6.12 / 6.56 |
