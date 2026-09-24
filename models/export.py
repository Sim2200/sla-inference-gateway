"""Export the candidate models to ONNX and build int8 variants.

Candidates
----------
resnet50_v1_fp32        ResNet-50, torchvision IMAGENET1K_V1 weights.
resnet50_v2_fp32        ResNet-50, IMAGENET1K_V2 weights (newer training recipe).
resnet50_v1_int8        Static int8 quantization of resnet50_v1_fp32.
resnet50_v2_int8        Static int8 quantization of resnet50_v2_fp32. The "accurate" tier;
                        resnet50_v1_int8 is the older version in the canary experiment.
resnet18_int8           Static int8 ResNet-18.
mobilenet_v3_large_fp32 MobileNetV3-Large, IMAGENET1K_V1. The "fast" tier.
mobilenet_v3_large_int8 Static int8 MobileNetV3-Large.

int8 models use ONNX Runtime static quantization (QDQ format, per-channel weights,
MinMax calibration on 500 held-out ImageNetV2 images, reduce_range because the
target CPU has AVX2 but no VNNI).

Each model is written to models/artifacts/<name>/model.onnx with a meta.json next
to it. The model server reads meta.json for the preprocessing sizes.

Run inside the builder image:  python models/export.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from torchvision import models as tvm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import split  # noqa: E402
from modelserver.preprocess import load_image, preprocess  # noqa: E402

ARTIFACTS = Path("models/artifacts")

FP32 = {
    # name: (constructor, weights enum, resize)
    "resnet50_v1_fp32": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V1, 256),
    "resnet50_v2_fp32": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2, 232),
    "resnet18_fp32": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1, 256),
    "mobilenet_v3_large_fp32": (tvm.mobilenet_v3_large, tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1, 256),
}
INT8 = {
    # name: fp32 source
    "resnet50_v1_int8": "resnet50_v1_fp32",
    "resnet50_v2_int8": "resnet50_v2_fp32",
    "resnet18_int8": "resnet18_fp32",
    "mobilenet_v3_large_int8": "mobilenet_v3_large_fp32",
}


def write_meta(name: str, **fields) -> None:
    path = ARTIFACTS / name / "model.onnx"
    meta = {"name": name, "size_mb": round(path.stat().st_size / 1e6, 2), **fields}
    (ARTIFACTS / name / "meta.json").write_text(json.dumps(meta, indent=2))


def export_fp32(name: str) -> None:
    ctor, weights, resize = FP32[name]
    model = ctor(weights=weights).eval()
    out = ARTIFACTS / name / "model.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, torch.randn(1, 3, 224, 224), str(out), dynamo=False, opset_version=17,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    onnx.checker.check_model(str(out))
    params = sum(p.numel() for p in model.parameters())
    write_meta(name, arch=ctor.__name__, weights=str(weights), precision="fp32",
               resize=resize, crop=224, params_millions=round(params / 1e6, 2))
    labels = ARTIFACTS / "labels.json"
    if not labels.exists():
        labels.write_text(json.dumps(weights.meta["categories"]))


class Calibration(CalibrationDataReader):
    def __init__(self, resize: int) -> None:
        calib, _ = split()
        self._batches = iter([{"input": preprocess(load_image(p.read_bytes()), resize)[None]}
                              for p, _ in calib])

    def get_next(self):
        return next(self._batches, None)


def export_int8(name: str) -> None:
    source = INT8[name]
    src_meta = json.loads((ARTIFACTS / source / "meta.json").read_text())
    out = ARTIFACTS / name / "model.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    prepped = out.with_name("prepped.onnx")
    quant_pre_process(str(ARTIFACTS / source / "model.onnx"), str(prepped))
    quantize_static(
        str(prepped), str(out), Calibration(src_meta["resize"]),
        quant_format=QuantFormat.QDQ, per_channel=True, reduce_range=True,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    prepped.unlink()
    write_meta(name, arch=src_meta["arch"], weights=src_meta["weights"], precision="int8",
               resize=src_meta["resize"], crop=224, params_millions=src_meta["params_millions"],
               quantization="static QDQ, per-channel, MinMax, 500 calibration images, reduce_range")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="subset of models to build (default: all)")
    args = parser.parse_args()
    names = args.names or [*FP32, *INT8]
    for name in names:
        start = time.perf_counter()
        (export_fp32 if name in FP32 else export_int8)(name)
        print(f"{name:26s} built in {time.perf_counter() - start:5.1f}s", flush=True)


if __name__ == "__main__":
    main()
