"""Measure accuracy and single-request latency of every built model.

accuracy: top-1 / top-5 over the 9,500 held-out ImageNetV2 images (all 1,000 classes).
latency:  batch-1 inference only (no decode/preprocess), ONNX Runtime pinned to the
          same thread count the model server uses. Run this mode in a container with
          the same CPU limit as a model server (see Makefile) so the numbers match.

Results are merged into results/models.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import split  # noqa: E402
from modelserver.preprocess import load_image, preprocess  # noqa: E402

ARTIFACTS = Path("models/artifacts")
RESULTS = Path("results/models.json")


def session(path: Path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def _prep(args: tuple[str, int]) -> np.ndarray:
    path, resize = args
    return preprocess(load_image(Path(path).read_bytes()), resize)


def accuracy(name: str, threads: int, limit: int | None, batch: int = 50) -> dict:
    meta = json.loads((ARTIFACTS / name / "meta.json").read_text())
    _, evaluation = split()
    evaluation = evaluation[:limit] if limit else evaluation
    sess = session(ARTIFACTS / name / "model.onnx", threads)
    labels = np.array([label for _, label in evaluation])
    top1 = top5 = 0
    with Pool(threads) as pool:
        for i in range(0, len(evaluation), batch):
            chunk = evaluation[i:i + batch]
            x = np.stack(pool.map(_prep, [(str(p), meta["resize"]) for p, _ in chunk]))
            logits = sess.run(None, {"input": x})[0]
            best5 = np.argsort(-logits, axis=1)[:, :5]
            y = labels[i:i + batch]
            top1 += int((best5[:, 0] == y).sum())
            top5 += int((best5 == y[:, None]).any(axis=1).sum())
    n = len(evaluation)
    return {"eval_images": n, "top1": round(top1 / n, 4), "top5": round(top5 / n, 4)}


def latency(name: str, threads: int, runs: int = 300, warmup: int = 30) -> dict:
    sess = session(ARTIFACTS / name / "model.onnx", threads)
    x = np.random.default_rng(0).standard_normal((1, 3, 224, 224)).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {"input": x})
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        sess.run(None, {"input": x})
        times.append((time.perf_counter() - start) * 1000)
    t = np.array(times)
    return {"latency_threads": threads, "latency_ms_p50": round(float(np.percentile(t, 50)), 2),
            "latency_ms_p95": round(float(np.percentile(t, 95)), 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["accuracy", "latency"])
    parser.add_argument("names", nargs="*")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--limit", type=int, help="evaluate on the first N images only (quick check)")
    args = parser.parse_args()

    names = args.names or sorted(p.parent.name for p in ARTIFACTS.glob("*/model.onnx"))
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for name in names:
        start = time.perf_counter()
        meta = json.loads((ARTIFACTS / name / "meta.json").read_text())
        row = results.setdefault(name, {})
        row.update({k: meta[k] for k in ("arch", "precision", "size_mb", "params_millions")})
        if args.mode == "accuracy":
            row.update(accuracy(name, args.threads, args.limit))
        else:
            row.update(latency(name, args.threads))
        print(f"{name:26s} {json.dumps(row)}  ({time.perf_counter() - start:.0f}s)", flush=True)
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(dict(sorted(results.items())), indent=2))


if __name__ == "__main__":
    main()
