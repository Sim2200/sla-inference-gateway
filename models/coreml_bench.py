"""Convert the gateway's ResNet-50 and MobileNetV3 models to Core ML and benchmark them on-device.

For each model this builds three Core ML packages:
  fp32   full precision
  fp16   16-bit weights and activations (the Core ML default)
  int8   fp16 model with weights linearly quantized to int8 (per-channel, symmetric)

and measures, on this Mac:
  - package size on disk
  - top-1 accuracy on the ImageNetV2 evaluation split (same split as the ONNX tiers)
  - single-image latency p50 / p95 per compute unit (CPU only, CPU + GPU)

Needs Python 3.12 with torch 2.2, torchvision 0.17 and coremltools (see Makefile target
`coreml`). Writes results/coreml.json and results/coreml.md.

    python models/coreml_bench.py --eval-images 2000 --latency-runs 200
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import torch
from PIL import Image
from torchvision import models as tvm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import split  # noqa: E402
from modelserver.preprocess import preprocess  # noqa: E402

OUT = Path("models/artifacts/coreml")
RESULTS = Path("results")

MODELS = {
    # name: (constructor, weights, resize)
    "resnet50_v2": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2, 232),
    "mobilenet_v3_large": (tvm.mobilenet_v3_large, tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1, 256),
}
COMPUTE_UNITS = {"cpu": ct.ComputeUnit.CPU_ONLY, "cpu_gpu": ct.ComputeUnit.CPU_AND_GPU}


def dir_size_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 2)


def convert(name: str) -> dict[str, Path]:
    ctor, weights, _ = MODELS[name]
    model = ctor(weights=weights).eval()
    traced = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
    inputs = [ct.TensorType(name="input", shape=(1, 3, 224, 224))]
    paths = {}
    for precision, ct_precision in (("fp32", ct.precision.FLOAT32), ("fp16", ct.precision.FLOAT16)):
        mlmodel = ct.convert(traced, inputs=inputs, convert_to="mlprogram",
                             compute_precision=ct_precision,
                             minimum_deployment_target=ct.target.macOS13)
        paths[precision] = save(mlmodel, name, precision)
        if precision == "fp16":
            config = cto.OptimizationConfig(
                global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                                                          granularity="per_channel"))
            paths["int8"] = save(cto.linear_quantize_weights(mlmodel, config), name, "int8")
    return paths


def save(mlmodel, name: str, precision: str) -> Path:
    path = OUT / f"{name}_{precision}.mlpackage"
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(path))
    return path


def output_key(mlmodel) -> str:
    return mlmodel.get_spec().description.output[0].name


def accuracy(path: Path, images: list[tuple[Path, int]], resize: int) -> float:
    mlmodel = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
    key = output_key(mlmodel)
    correct = 0
    for img_path, label in images:
        x = preprocess(Image.open(img_path).convert("RGB"), resize)[None]
        logits = mlmodel.predict({"input": x})[key]
        correct += int(np.argmax(logits) == label)
    return correct / len(images)


def latency(path: Path, unit, runs: int, sample: np.ndarray) -> dict[str, float]:
    mlmodel = ct.models.MLModel(str(path), compute_units=unit)
    for _ in range(20):  # warm-up
        mlmodel.predict({"input": sample})
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        mlmodel.predict({"input": sample})
        times.append((time.perf_counter() - start) * 1000)
    return {"p50_ms": round(float(np.percentile(times, 50)), 2),
            "p95_ms": round(float(np.percentile(times, 95)), 2)}


def hardware() -> str:
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                         capture_output=True, text=True).stdout.strip()
    return f"{cpu}, macOS {platform.mac_ver()[0]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-images", type=int, default=2000)
    parser.add_argument("--latency-runs", type=int, default=200)
    args = parser.parse_args()

    _, evaluation = split()
    images = evaluation[: args.eval_images]
    rows = []
    for name, (_, _, resize) in MODELS.items():
        print(f"converting {name}", flush=True)
        paths = convert(name)
        sample = preprocess(Image.open(images[0][0]).convert("RGB"), resize)[None]
        for precision, path in paths.items():
            row = {"model": name, "precision": precision, "size_mb": dir_size_mb(path),
                   "top1": round(accuracy(path, images, resize), 4)}
            for unit_name, unit in COMPUTE_UNITS.items():
                row[unit_name] = latency(path, unit, args.latency_runs, sample)
            print(json.dumps(row), flush=True)
            rows.append(row)

    RESULTS.mkdir(exist_ok=True)
    meta = {"hardware": hardware(), "eval_images": len(images),
            "latency_runs": args.latency_runs, "coremltools": ct.__version__, "rows": rows}
    (RESULTS / "coreml.json").write_text(json.dumps(meta, indent=2))
    lines = [f"# Core ML on-device benchmark\n",
             f"Hardware: {meta['hardware']}. Top-1 on {len(images)} ImageNetV2 evaluation images; "
             f"latency is single-image predict over {args.latency_runs} runs after warm-up.\n",
             "| Model | Precision | Size (MB) | Top-1 | CPU p50 / p95 (ms) | CPU+GPU p50 / p95 (ms) |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['precision']} | {r['size_mb']} | {r['top1']:.2%} | "
                     f"{r['cpu']['p50_ms']} / {r['cpu']['p95_ms']} | "
                     f"{r['cpu_gpu']['p50_ms']} / {r['cpu_gpu']['p95_ms']} |")
    (RESULTS / "coreml.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
