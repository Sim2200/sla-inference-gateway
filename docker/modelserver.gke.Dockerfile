# Model server with the served models baked in (no host volume on GKE).
FROM slagw-modelserver
COPY models/artifacts/labels.json /models/labels.json
COPY models/artifacts/resnet50_v2_int8 /models/resnet50_v2_int8
COPY models/artifacts/mobilenet_v3_large_fp32 /models/mobilenet_v3_large_fp32
COPY models/artifacts/resnet50_v1_int8 /models/resnet50_v1_int8
