"""The Triton comparison on one Kaggle T4. Pushed by scripts/run_on_kaggle.py.

Reads the project from the attached private dataset (sla-gateway-src: src/, loadtest/, deploy/,
two ONNX models, a 500-image replay set), installs the serving stack, runs
loadtest/triton_experiment.py and writes results/triton_load.json plus the raw logs.

Two interpreters: the image's Python runs the gateway, the FastAPI model servers and the load
generator; Triton's Python backend needs an interpreter whose standard library is importable by
its stub, which the image's Python is not, so PyTriton is installed into a conda-managed 3.12.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
SRC = Path(glob.glob("/kaggle/input/**/src/gateway/app.py", recursive=True)[0]).parents[2]
PROJ = WORK / "proj"
PY = sys.executable
PY312 = Path("/tmp/conda312/bin/python")  # outside /kaggle/working so it is not part of the kernel output


def sh(*args, check=True, capture=False, env=None):
    print("$", " ".join(str(a) for a in args)[:300], flush=True)
    t0 = time.time()
    r = subprocess.run([str(a) for a in args], text=True, capture_output=capture, env={**os.environ, **(env or {})}, cwd=PROJ)
    print(f"  -> exit {r.returncode} in {time.time() - t0:.0f} s", flush=True)
    if check and r.returncode != 0:
        if capture:
            print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"step failed: {args[0]}")
    return r


def main():
    # a writable copy of the project (the dataset mount is read-only)
    shutil.copytree(SRC, PROJ, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    (PROJ / "results/raw").mkdir(parents=True, exist_ok=True)

    sh(PY, "-m", "pip", "install", "-q", "fastapi", "uvicorn[standard]", "httpx", "pyyaml", "prometheus-client", "opentelemetry-sdk",
       "opentelemetry-instrumentation-fastapi", "opentelemetry-instrumentation-httpx", "opentelemetry-exporter-otlp-proto-http",
       "onnxruntime-gpu==1.22.0", "pillow", "numpy", check=False)  # 1.22: the last build for CUDA 12 on PyPI
    conda = Path("/opt/conda/bin/conda")
    if conda.exists():
        sh(conda, "create", "-y", "-q", "-p", "/tmp/conda312", "python=3.12", check=False)
    else:
        sh("bash", "-c", "curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /tmp bin/micromamba && "
           f"/tmp/bin/micromamba create -y -q -r /tmp/mm -p /tmp/conda312 -c conda-forge python=3.12", check=False)
    sh(PY312, "-m", "pip", "install", "-q", "nvidia-pytriton", "onnxruntime-gpu==1.22.0", "numpy", "pillow", check=False)
    env = {"gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
           "python_serving": sys.version.split()[0],
           "python_triton": subprocess.run([str(PY312), "-c", "import sys; print(sys.version.split()[0])"], capture_output=True, text=True).stdout.strip(),
           "pytriton": subprocess.run([str(PY312), "-c", "import pytriton; print(pytriton.__version__)"], capture_output=True, text=True).stdout.strip(),
           "onnxruntime_gpu": subprocess.run([str(PY), "-c", "import onnxruntime as o; print(o.__version__)"], capture_output=True, text=True).stdout.strip(),
           "cpus": os.cpu_count()}
    print(env, flush=True)

    sh(PY, "loadtest/triton_experiment.py", "--replay-dir", "data/replay", "--python", PY, "--python-triton", PY312,
       "--out", "results/triton_load.json", check=False)

    out = PROJ / "results/triton_load.json"
    if out.exists():
        d = json.loads(out.read_text())
        d["env"] = {**d.get("env", {}), **env}
        out.write_text(json.dumps(d, indent=2))
        print(out.read_text()[:3000], flush=True)
    # expose results at the kernel output root
    for f in (PROJ / "results").rglob("*"):
        if f.is_file() and f.suffix in (".json", ".log"):
            shutil.copy(f, WORK / f.name)


if __name__ == "__main__":
    main()
