"""Energy per inference on this Mac, measured with powermetrics (macOS only, needs sudo).

For every model version the script measures, in this order and three times over:
  1. an idle baseline: powermetrics sampling while nothing runs,
  2. a work window: the same sampling while the model runs back to back, counting inferences.
Energy per inference = (mean work power - mean idle power) x window seconds / inferences.
Only what powermetrics reports is used. On Intel Macs the `cpu_power` sampler gives the
package power (CPU cores + integrated GPU + system agent) and the average core frequency;
whether a separate GPU power figure exists is recorded per run rather than assumed
(a discrete AMD GPU is not covered by powermetrics). Nothing is estimated.

A final 5-minute sustained run on one model records power, frequency and throughput per
10-second bucket to show whether the machine throttles. Output: results/power.json.

    sudo -v && .venv-coreml/bin/python models/power_bench.py [--window 20] [--repeats 3] [--sustain 300]
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import onnxruntime as ort

COREML = Path("models/artifacts/coreml")
ONNX = Path("models/artifacts")
RESULTS = Path("results")

COREML_CONFIGS = [(m, p, u) for m in ("resnet50_v2", "mobilenet_v3_large") for p in ("fp32", "fp16", "int8")
                  for u in ("cpu", "cpu_gpu")]
ONNX_CONFIGS = [("resnet50_v2", "fp32"), ("resnet50_v2", "int8"), ("mobilenet_v3_large", "fp32"),
                ("mobilenet_v3_large", "int8")]
UNITS = {"cpu": ct.ComputeUnit.CPU_ONLY, "cpu_gpu": ct.ComputeUnit.CPU_AND_GPU}

PKG_RE = re.compile(r"package power[^:]*:\s*([\d.]+)\s*W", re.I)
GPU_RE = re.compile(r"GPU Power:\s*([\d.]+)\s*mW", re.I)
FREQ_RE = re.compile(r"System Average frequency as fraction of nominal:\s*[\d.]+%\s*\(([\d.]+)\s*MHz\)", re.I)


# ---------------------------------------------------------------- powermetrics

class Sampler:
    """Runs powermetrics for the duration of a `with` block and parses its output afterwards."""

    def __init__(self, interval_ms: int = 500):
        self.interval_ms = interval_ms
        self.file = Path(tempfile.mkstemp(suffix=".pm.txt")[1])

    def __enter__(self):
        self.proc = subprocess.Popen(
            ["sudo", "-n", "powermetrics", "--samplers", "cpu_power,gpu_power", "-i", str(self.interval_ms),
             "--output-file", str(self.file)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        time.sleep(0.5)
        if self.proc.poll() is not None:
            raise SystemExit("powermetrics did not start: " + self.proc.stderr.read().decode())
        return self

    def __exit__(self, *exc):
        subprocess.run(["sudo", "-n", "kill", "-INT", str(self.proc.pid)], check=False)
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            subprocess.run(["sudo", "-n", "kill", "-KILL", str(self.proc.pid)], check=False)
        text = self.file.read_text(errors="replace")
        self.file.unlink(missing_ok=True)
        self.package_w = [float(m) for m in PKG_RE.findall(text)]
        self.gpu_mw = [float(m) for m in GPU_RE.findall(text)]
        self.freq_mhz = [float(m) for m in FREQ_RE.findall(text)]

    def summary(self) -> dict:
        def stats(xs):
            return {"mean": round(statistics.fmean(xs), 3), "min": round(min(xs), 3), "max": round(max(xs), 3),
                    "samples": len(xs)} if xs else None
        return {"package_w": stats(self.package_w), "gpu_mw": stats(self.gpu_mw), "freq_mhz": stats(self.freq_mhz)}


# ---------------------------------------------------------------- models

def coreml_runner(name: str, precision: str, unit: str):
    m = ct.models.MLModel(str(COREML / f"{name}_{precision}.mlpackage"), compute_units=UNITS[unit])
    key = m.get_spec().description.output[0].name
    x = np.random.rand(1, 3, 224, 224).astype(np.float32)
    return lambda: m.predict({"input": x})[key]


def onnx_runner(name: str, precision: str, threads: int):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    s = ort.InferenceSession(str(ONNX / f"{name}_{precision}" / "model.onnx"), so, providers=["CPUExecutionProvider"])
    inp = s.get_inputs()[0].name
    x = np.random.rand(1, 3, 224, 224).astype(np.float32)
    return lambda: s.run(None, {inp: x})


def run_for(fn, seconds: float) -> tuple[int, list[float]]:
    n, lat, end = 0, [], time.perf_counter() + seconds
    while time.perf_counter() < end:
        t0 = time.perf_counter()
        fn()
        lat.append((time.perf_counter() - t0) * 1000)
        n += 1
    return n, lat


def measure(label: str, fn, window: float, repeats: int, idle_seconds: float) -> dict:
    for _ in range(15):  # warm-up, outside any sampling window
        fn()
    runs = []
    for r in range(repeats):
        time.sleep(3)  # let the package settle before the idle sample
        with Sampler() as idle:
            time.sleep(idle_seconds)
        with Sampler() as work:
            n, lat = run_for(fn, window)
        i, w = idle.summary(), work.summary()
        if not (i["package_w"] and w["package_w"]):
            raise SystemExit(f"{label}: powermetrics reported no package power; nothing to compute")
        delta_w = w["package_w"]["mean"] - i["package_w"]["mean"]
        run = {"repeat": r, "inferences": n, "window_s": window, "throughput_per_s": round(n / window, 2),
               "p50_ms": round(float(np.percentile(lat, 50)), 2), "idle": i, "work": w,
               "delta_package_w": round(delta_w, 3), "mj_per_inference": round(delta_w * window / n * 1000, 2),
               "package_mj_per_inference_incl_idle": round(w["package_w"]["mean"] * window / n * 1000, 2)}
        if i["gpu_mw"] and w["gpu_mw"]:
            run["gpu_mj_per_inference"] = round((w["gpu_mw"]["mean"] - i["gpu_mw"]["mean"]) / 1000 * window / n * 1000, 2)
        runs.append(run)
        print(f"  {label:38s} run {r}: {n:5d} inf, idle {i['package_w']['mean']:.2f} W, work {w['package_w']['mean']:.2f} W,"
              f" {run['mj_per_inference']:.1f} mJ/inf, freq {w['freq_mhz']['mean'] if w['freq_mhz'] else '-'} MHz", flush=True)
    mj = [x["mj_per_inference"] for x in runs]
    return {"label": label, "runs": runs, "mj_per_inference": {"median": round(statistics.median(mj), 2),
                                                               "min": round(min(mj), 2), "max": round(max(mj), 2)},
            "throughput_per_s": round(statistics.median(x["throughput_per_s"] for x in runs), 2),
            "p50_ms": round(statistics.median(x["p50_ms"] for x in runs), 2),
            "gpu_power_reported": all("gpu_mj_per_inference" in x for x in runs)}


def sustained(fn, seconds: float, bucket: float = 10.0) -> dict:
    """One long run: per-bucket throughput, with power and frequency from one continuous sample."""
    for _ in range(15):
        fn()
    buckets = []
    with Sampler(interval_ms=1000) as s:
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < seconds:
            n, lat = run_for(fn, bucket)
            buckets.append({"t": round(time.perf_counter() - t_start), "throughput_per_s": round(n / bucket, 2),
                            "p50_ms": round(float(np.percentile(lat, 50)), 2)})
    # Align 1-second power/frequency samples to the buckets.
    per = max(1, int(bucket))
    for i, b in enumerate(buckets):
        pw, fq = s.package_w[i * per:(i + 1) * per], s.freq_mhz[i * per:(i + 1) * per]
        b["package_w"] = round(statistics.fmean(pw), 2) if pw else None
        b["freq_mhz"] = round(statistics.fmean(fq), 0) if fq else None
    first, last = buckets[:3], buckets[-3:]
    mean = lambda xs, k: statistics.fmean(x[k] for x in xs if x[k] is not None) if any(x[k] is not None for x in xs) else None  # noqa: E731
    return {"seconds": seconds, "bucket_s": bucket, "buckets": buckets, "overall": s.summary(),
            "first_30s": {k: mean(first, k) for k in ("throughput_per_s", "package_w", "freq_mhz")},
            "last_30s": {k: mean(last, k) for k in ("throughput_per_s", "package_w", "freq_mhz")}}


def hardware() -> dict:
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    gpus = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True).stdout
    return {"cpu": cpu, "macos": platform.mac_ver()[0], "gpus": re.findall(r"Chipset Model:\s*(.+)", gpus),
            "coremltools": ct.__version__, "onnxruntime": ort.__version__}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=float, default=20, help="seconds of back-to-back inference per repeat")
    ap.add_argument("--idle", type=float, default=10, help="seconds of idle baseline per repeat")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--sustain", type=float, default=300, help="seconds for the throttling run (0 to skip)")
    ap.add_argument("--sustain-model", default="coreml:resnet50_v2:fp32:cpu")
    ap.add_argument("--onnx-threads", type=int, default=4)
    ap.add_argument("--only", default="", help="substring filter on config labels (for quick checks)")
    a = ap.parse_args()
    probe = subprocess.run(["sudo", "-n", "powermetrics", "--samplers", "cpu_power", "-i", "100", "-n", "1"],
                           capture_output=True)
    if probe.returncode != 0:
        raise SystemExit("powermetrics needs administrator access: run `sudo -v` first, or allow it in sudoers")

    configs = [(f"coreml:{m}:{p}:{u}", lambda m=m, p=p, u=u: coreml_runner(m, p, u)) for m, p, u in COREML_CONFIGS]
    configs += [(f"onnxruntime:{m}:{p}:cpu{a.onnx_threads}", lambda m=m, p=p: onnx_runner(m, p, a.onnx_threads))
                for m, p in ONNX_CONFIGS]
    configs = [c for c in configs if a.only in c[0]]
    print(f"{len(configs)} configurations x {a.repeats} repeats, {a.window}s windows", flush=True)
    rows = [measure(label, make(), a.window, a.repeats, a.idle) for label, make in configs]
    out = {"hardware": hardware(), "method": {"window_s": a.window, "idle_s": a.idle, "repeats": a.repeats,
                                              "powermetrics_samplers": "cpu_power,gpu_power", "interval_ms": 500,
                                              "input": "random 1x3x224x224 tensor, batch 1"},
           "rows": rows, "gpu_power_reported": any(r["gpu_power_reported"] for r in rows)}
    if a.sustain > 0:
        kind, m, p, u = a.sustain_model.split(":")
        fn = coreml_runner(m, p, u) if kind == "coreml" else onnx_runner(m, p, a.onnx_threads)
        print(f"sustained {a.sustain}s on {a.sustain_model}", flush=True)
        out["sustained"] = {"model": a.sustain_model, **sustained(fn, a.sustain)}
        f, l = out["sustained"]["first_30s"], out["sustained"]["last_30s"]
        print(f"  first 30 s: {f}\n  last 30 s:  {l}", flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "power.json").write_text(json.dumps(out, indent=2))
    print("wrote results/power.json")


if __name__ == "__main__":
    main()
