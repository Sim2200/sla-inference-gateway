"""Serve one of the project's ONNX classifiers from Triton Inference Server with dynamic batching.

    python -m triton.serve --model-dir models/artifacts/resnet50_v2_int8 --name resnet50_v2_int8 \
        --max-batch 32 --queue-delay-us 5000 [--no-batching] [--provider CUDAExecutionProvider]

Triton runs in-process through PyTriton, which bundles the `tritonserver` binary and exposes the
model through Triton's Python backend; the HTTP (:8000), gRPC (:8001) and metrics (:8002) endpoints
are the real Triton ones, and the dynamic batcher is Triton's. The request contract matches what
the gateway sends (src/gateway/app.py, `_call_triton`): one BYTES element holding the base64 image,
one (1000,) FP32 logits output. Preprocessing (decode, resize, crop, normalise) is the same code
the FastAPI model server uses, so the two tiers answer identically for the same model file.

Why in-process rather than the Triton container: the free GPU boxes used for the measurements
(Kaggle) have no Docker. The honest consequence is that the model runs through Triton's Python
backend calling ONNX Runtime, not Triton's native onnxruntime backend; batching, queueing, the
protocol and the metrics are Triton's. `--dump-config` writes the config Triton actually loaded,
the equivalent of a hand-written config.pbtxt.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modelserver.preprocess import preprocess_bytes  # noqa: E402


def main() -> None:
    from pytriton.decorators import batch
    from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
    from pytriton.triton import Triton, TritonConfig

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--queue-delay-us", type=int, default=5000, help="how long the batcher waits to fill a batch")
    ap.add_argument("--no-batching", action="store_true", help="max batch 1, no dynamic batcher (the control arm)")
    ap.add_argument("--provider", default="CUDAExecutionProvider")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--http-port", type=int, default=8000)
    ap.add_argument("--grpc-port", type=int, default=8001)
    ap.add_argument("--metrics-port", type=int, default=8002)
    ap.add_argument("--dump-config", default="", help="write the loaded Triton model config (JSON) here")
    a = ap.parse_args()

    model_dir = Path(a.model_dir)
    meta = json.loads((model_dir / "meta.json").read_text())
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = a.threads
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    sess = ort.InferenceSession(str(model_dir / "model.onnx"), opts, providers=[a.provider, "CPUExecutionProvider"])
    print("providers", sess.get_providers(), flush=True)
    x = np.zeros((1, 3, meta["crop"], meta["crop"]), dtype=np.float32)
    for _ in range(5):
        sess.run(None, {"input": x})

    @batch
    def infer(image: np.ndarray) -> dict:
        # `image` is (N, 1) of bytes: N requests Triton's batcher put together. Decode each, then
        # one ONNX Runtime call for the whole batch; that batching is where the GPU wins.
        xs = [preprocess_bytes(base64.b64decode(row[0]), meta["resize"], meta["crop"])[0] for row in image]
        logits = sess.run(None, {"input": np.stack(xs).astype(np.float32)})[0]
        return {"logits": logits.astype(np.float32)}

    if a.no_batching:
        config = ModelConfig(max_batch_size=1, batching=False)
    else:
        config = ModelConfig(max_batch_size=a.max_batch, batcher=DynamicBatcher(max_queue_delay_microseconds=a.queue_delay_us))

    triton_cfg = TritonConfig(http_port=a.http_port, grpc_port=a.grpc_port, metrics_port=a.metrics_port, log_verbose=0)
    with Triton(config=triton_cfg) as triton:
        triton.bind(model_name=a.name, infer_func=infer,
                    inputs=[Tensor(name="image", dtype=np.bytes_, shape=(1,))],
                    outputs=[Tensor(name="logits", dtype=np.float32, shape=(1000,))],
                    config=config, strict=False)
        if a.dump_config:
            import urllib.request

            time.sleep(1)
            with urllib.request.urlopen(f"http://127.0.0.1:{a.http_port}/v2/models/{a.name}/config", timeout=10) as r:
                Path(a.dump_config).write_text(json.dumps(json.loads(r.read()), indent=2))
        print(f"SERVING {a.name} max_batch={config.max_batch_size} batching={'off' if a.no_batching else 'dynamic'} "
              f"pid={os.getpid()}", flush=True)
        triton.serve()


if __name__ == "__main__":
    main()
