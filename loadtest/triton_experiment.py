"""Triton vs the FastAPI model server, on one GPU box. Writes results/triton_load.json.

    python loadtest/triton_experiment.py --replay-dir data/replay --python-triton /path/to/py312 \
        --out results/triton_load.json

Four ways of serving the same ONNX file (ResNet-50 int8, the gateway's accurate tier), each
measured through the gateway in `always_accurate` mode with the open-loop load generator:

    fastapi_cpu      src/modelserver as shipped: ONNX Runtime CPU, 2 threads, one inference at a time
    fastapi_gpu      the same server with ORT_PROVIDER=CUDAExecutionProvider (no batching)
    triton_nobatch   Triton (PyTriton, Python backend -> ONNX Runtime CUDA), max batch 1
    triton_batch     Triton with dynamic batching: max batch --max-batch, queue delay --queue-delay-us

For each arm: a capacity sweep (offered load up until p95 crosses the SLA or errors appear), then
runs at 1x and 2x a common reference load (the fastapi_gpu capacity, so every arm sees the same
requests per second). Triton's own metrics give the batch sizes it actually formed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLA_MS = 300
GATEWAY = "http://127.0.0.1:8080"


def http(url: str, payload: dict | None = None, timeout: float = 10) -> dict | str:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    try:
        return json.loads(body)
    except ValueError:
        return body


def wait_http(url: str, timeout: float = 240) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=3)
            return
        except Exception:  # noqa: BLE001
            time.sleep(1)
    raise SystemExit(f"timeout waiting for {url}")


class Proc:
    def __init__(self, name: str, cmd: list[str], env: dict | None = None, log_dir: Path = Path("results/raw")):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log = open(log_dir / f"triton_{name}.log", "w")
        self.p = subprocess.Popen(cmd, env={**os.environ, **(env or {})}, stdout=self.log, stderr=subprocess.STDOUT,
                                  cwd=ROOT, start_new_session=True)

    def stop(self) -> None:
        if self.p.poll() is None:
            os.killpg(self.p.pid, signal.SIGTERM)
            try:
                self.p.wait(15)
            except subprocess.TimeoutExpired:
                os.killpg(self.p.pid, signal.SIGKILL)
        self.log.close()


def triton_env(python: str) -> dict:
    """Triton's Python backend runs the model in its own stub process, which embeds the interpreter
    but does not inherit a virtual or conda environment's module path unless that environment is
    activated. Activate it the explicit way: its site-packages on PYTHONPATH, its lib/ on
    LD_LIBRARY_PATH, its bin/ first on PATH."""
    q = lambda code: subprocess.run([python, "-c", code], capture_output=True, text=True).stdout.strip()  # noqa: E731
    site = q("import site; print(site.getsitepackages()[0])")
    prefix = q("import sys; print(sys.prefix)")
    return {"PYTHONPATH": site, "LD_LIBRARY_PATH": f"{prefix}/lib:" + os.environ.get("LD_LIBRARY_PATH", ""),
            "PATH": f"{prefix}/bin:" + os.environ.get("PATH", ""), "CONDA_PREFIX": prefix, "VIRTUAL_ENV": prefix}


def triton_metrics() -> dict:
    """Batch statistics from Triton's Prometheus endpoint."""
    try:
        text = http("http://127.0.0.1:8002/metrics")
    except Exception:  # noqa: BLE001
        return {}
    val = lambda name: sum(float(m.group(1)) for m in re.finditer(rf'^{name}\{{[^}}]*\}} ([0-9.e+]+)$', text, re.M))  # noqa: E731
    reqs, execs = val("nv_inference_request_success"), val("nv_inference_exec_count")
    return {"requests": reqs, "executions": execs, "mean_batch_size": round(reqs / execs, 2) if execs else None,
            "queue_us_per_request": round(val("nv_inference_queue_duration_us") / reqs, 1) if reqs else None,
            "compute_us_per_request": round(val("nv_inference_compute_infer_duration_us") / reqs, 1) if reqs else None}


def loadgen(profile: str, out: str, replay: str, py: str, warmup_s: float = 5, timeout: float = 30) -> dict:
    cmd = [py, "loadtest/loadgen.py", "--url", GATEWAY, "--profile", profile, "--out", f"results/raw/{out}",
           "--sla-ms", str(SLA_MS), "--warmup-s", str(warmup_s), "--timeout", str(timeout), "--images", "500",
           "--replay-dir", replay]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"loadgen failed ({r.returncode}): {r.stderr[-1500:]}")
    return json.loads((ROOT / "results/raw" / f"{out}.json").read_text())["overall"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir", required=True)
    ap.add_argument("--python-triton", default=sys.executable, help="interpreter with nvidia-pytriton installed")
    ap.add_argument("--python", default=sys.executable, help="interpreter for the gateway, model server and load generator")
    ap.add_argument("--model-dir", default="models/artifacts/resnet50_v2_int8")
    ap.add_argument("--fast-model-dir", default="models/artifacts/mobilenet_v3_large_fp32")
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--queue-delay-us", type=int, default=5000)
    ap.add_argument("--rates", default="10,20,40,60,80,120,160,240,320")
    ap.add_argument("--point-s", type=int, default=30)
    ap.add_argument("--run-s", type=int, default=60)
    ap.add_argument("--arms", default="fastapi_gpu,fastapi_cpu,triton_nobatch,triton_batch",
                    help="fastapi_gpu first: its capacity is the 1x reference load for every arm")
    ap.add_argument("--out", default="results/triton_load.json")
    a = ap.parse_args()
    rates = [float(r) for r in a.rates.split(",")]
    model_name = Path(a.model_dir).name

    arms = {
        "fastapi_cpu": dict(kind="modelserver", env={"ORT_PROVIDER": "CPUExecutionProvider"}),
        "fastapi_gpu": dict(kind="modelserver", env={"ORT_PROVIDER": "CUDAExecutionProvider"}),
        "triton_nobatch": dict(kind="triton", args=["--no-batching"]),
        "triton_batch": dict(kind="triton", args=["--max-batch", str(a.max_batch), "--queue-delay-us", str(a.queue_delay_us)]),
    }
    out = {"sla_ms": SLA_MS, "model": model_name, "arms": {}, "rates_swept": rates, "point_s": a.point_s, "run_s": a.run_s,
           "triton": {"max_batch": a.max_batch, "queue_delay_us": a.queue_delay_us}}
    env_info = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True)
    out["env"] = {"gpu": env_info.stdout.strip(), "cpus": os.cpu_count()}

    # The fast tier is never used in always_accurate mode but the gateway needs one configured.
    fast = Proc("fast", [a.python, "-m", "uvicorn", "modelserver.app:app", "--host", "127.0.0.1", "--port", "8101", "--no-access-log"],
                env={"MODEL_DIR": str(ROOT / a.fast_model_dir), "TIER": "fast", "PYTHONPATH": str(ROOT / "src")})
    wait_http("http://127.0.0.1:8101/readyz")
    reference_rps: float | None = None
    try:
        for arm in a.arms.split(","):
            spec = arms[arm]
            try:
                reference_rps = run_arm(a, arm, spec, model_name, out, reference_rps, rates)
            except Exception as exc:  # noqa: BLE001 - one broken arm must not lose the others' numbers
                out["arms"][arm] = {"error": f"{type(exc).__name__}: {exc}"}
                print(arm, "FAILED", exc, flush=True)
            finally:
                for proc in PROCS:
                    proc.stop()
                PROCS.clear()
                time.sleep(3)
    finally:
        fast.stop()
    out["reference_rps"] = reference_rps
    out["note"] = ("One GPU box, every process on the same machine (load generator included); fastapi_cpu is the tier as "
                   "shipped (CPU). The Triton arms use Triton's Python backend calling ONNX Runtime CUDA, not the native "
                   "onnxruntime backend, because the free GPU host has no Docker for the Triton container.")
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)


PROCS: list[Proc] = []  # everything started for the current arm; stopped after it


def run_arm(a, arm: str, spec: dict, model_name: str, out: dict, reference_rps: float | None, rates: list[float]) -> float | None:
    if spec["kind"] == "modelserver":
        backend = Proc(arm, [a.python, "-m", "uvicorn", "modelserver.app:app", "--host", "127.0.0.1", "--port", "8100", "--no-access-log"],
                       env={"MODEL_DIR": str(ROOT / a.model_dir), "TIER": "accurate", "PYTHONPATH": str(ROOT / "src"), **spec["env"]})
        PROCS.append(backend)
        wait_http("http://127.0.0.1:8100/readyz")
        info = http("http://127.0.0.1:8100/info")
        out["arms"].setdefault(arm, {})["serving"] = {k: info.get(k) for k in ("ort_provider", "ort_providers_active", "onnxruntime", "ort_threads", "max_concurrency")}
        gw_env = {"ACCURATE_URL": "http://127.0.0.1:8100", "ACCURATE_NAME": arm}
    else:
        tenv = triton_env(a.python_triton)
        backend = Proc(arm, [a.python_triton, "-m", "triton.serve", "--model-dir", a.model_dir, "--name", model_name,
                             "--dump-config", f"results/raw/triton_{arm}_config.json", *spec["args"]],
                       env={**tenv, "PYTHONPATH": f"{ROOT / 'src'}:{tenv['PYTHONPATH']}"})
        PROCS.append(backend)
        wait_http(f"http://127.0.0.1:8000/v2/models/{model_name}/ready")
        gw_env = {"ACCURATE_URL": "http://127.0.0.1:8000", "ACCURATE_NAME": arm, "ACCURATE_PROTOCOL": "triton",
                  "ACCURATE_MODEL": model_name}
    gateway = Proc(f"gateway_{arm}", [a.python, "-m", "uvicorn", "gateway.app:create_app", "--factory", "--host", "127.0.0.1",
                                      "--port", "8080", "--no-access-log"],
                   env={"GATEWAY_CONFIG": str(ROOT / "deploy/registry.yaml"), "GATEWAY_MODE": "always_accurate",
                        "FAST_URL": "http://127.0.0.1:8101", "PYTHONPATH": str(ROOT / "src"), **gw_env})
    PROCS.append(gateway)
    wait_http(f"{GATEWAY}/healthz")
    loadgen("5:8", f"triton_{arm}_warm", a.replay_dir, a.python)  # warm the whole path
    record = {"capacity_sweep": [], "capacity_rps": None, "runs": {}}

    # 1. capacity: climb until the SLA breaks
    for rps in rates:
        before = triton_metrics()
        s = loadgen(f"{rps}:{a.point_s}", f"triton_{arm}_cap_{int(rps)}", a.replay_dir, a.python)
        after = triton_metrics()
        point = {"rps": rps, **{k: s[k] for k in ("offered_rps", "goodput_rps", "p50_ms", "p95_ms", "p99_ms", "within_sla", "error_rate")}}
        if after and before:
            d_req, d_exec = after["requests"] - before["requests"], after["executions"] - before["executions"]
            point["triton_mean_batch"] = round(d_req / d_exec, 2) if d_exec else None
        record["capacity_sweep"].append(point)
        print(arm, "capacity", point, flush=True)
        if s["within_sla"] < 0.9 or s["error_rate"] > 0.05:
            break
        record["capacity_rps"] = rps
    if arm == "fastapi_gpu":
        reference_rps = record["capacity_rps"] or rates[0]

    # 2. 1x and 2x the reference load (set once the reference arm has run)
    if reference_rps is not None:
        for mult in (1, 2):
            rps = round(reference_rps * mult, 1)
            before = triton_metrics()
            s = loadgen(f"{rps}:{a.run_s}", f"triton_{arm}_x{mult}", a.replay_dir, a.python, warmup_s=10)
            after = triton_metrics()
            run = {"rps": rps, **s}
            if after and before:
                d_req, d_exec = after["requests"] - before["requests"], after["executions"] - before["executions"]
                run["triton_mean_batch"] = round(d_req / d_exec, 2) if d_exec else None
                run["triton_queue_us_per_request"] = round((after.get("queue_us_per_request") or 0), 1)
            record["runs"][f"{mult}x"] = run
            print(arm, f"{mult}x", {k: run[k] for k in ("rps", "p50_ms", "p95_ms", "goodput_rps", "error_rate")}, flush=True)
    cfg = ROOT / f"results/raw/triton_{arm}_config.json"
    if cfg.exists():
        record["triton_model_config"] = json.loads(cfg.read_text())
        record["serving"] = record["triton_model_config"].pop("_serving", None)
    record.setdefault("serving", out["arms"].get(arm, {}).get("serving"))
    out["arms"][arm] = record
    return reference_rps


if __name__ == "__main__":
    main()
