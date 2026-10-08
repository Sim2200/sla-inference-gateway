"""Run the Triton comparison on Kaggle's free T4 and pull the results back.

1. Packages the source, the two ONNX models and a 500-image replay set into a private Kaggle
   dataset (<user>/sla-gateway-src); kernels can attach datasets but not git repositories.
2. Pushes kaggle/run_triton.py as a private GPU script kernel with internet access (for the
   pip wheels), polls until it finishes, downloads results/triton_load.json and the raw logs.

    python scripts/run_on_kaggle.py --user <kaggle username>
    python scripts/run_on_kaggle.py --user <kaggle username> --pull-only
Needs the Kaggle CLI authenticated (~/.kaggle/kaggle.json or ~/.kaggle/access_token).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = [sys.executable, "-m", "kaggle"]
MODELS = ("resnet50_v2_int8", "mobilenet_v3_large_fp32")
REPLAY_IMAGES = 500


def kaggle(*args: str, check: bool = True) -> str:
    r = subprocess.run([*KAGGLE, *args], text=True, capture_output=True)
    if check and r.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def push_source(user: str) -> str:
    slug = f"{user}/sla-gateway-src"
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for sub in ("src", "loadtest", "deploy"):
            shutil.copytree(ROOT / sub, d / sub, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for m in MODELS:
            shutil.copytree(ROOT / "models/artifacts" / m, d / "models/artifacts" / m)
        shutil.copy(ROOT / "models/artifacts/labels.json", d / "models/artifacts/labels.json")
        labels = json.loads((ROOT / "data/replay/labels.json").read_text())
        keep = sorted(labels)[:REPLAY_IMAGES]
        (d / "data/replay").mkdir(parents=True)
        for name in keep:
            shutil.copy(ROOT / "data/replay" / name, d / "data/replay" / name)
        (d / "data/replay/labels.json").write_text(json.dumps({k: labels[k] for k in keep}))
        (d / "dataset-metadata.json").write_text(json.dumps({"title": "sla-gateway-src", "id": slug, "licenses": [{"name": "CC0-1.0"}]}))
        exists = slug.split("/")[1] in kaggle("datasets", "list", "--mine", check=False)
        out = kaggle("datasets", "version" if exists else "create", "-p", str(d), *(["-m", "update"] if exists else []), "--dir-mode", "zip", check=False)
        print(out.strip().splitlines()[-1])
    time.sleep(20)
    return slug


def push_kernel(user: str, src_slug: str) -> str:
    kid = f"{user}/sla-gateway-triton"
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        shutil.copy(ROOT / "kaggle" / "run_triton.py", d / "run_triton.py")
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": kid, "title": "sla-gateway-triton", "code_file": "run_triton.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [src_slug], "competition_sources": [], "kernel_sources": []}))
        print(kaggle("kernels", "push", "-p", str(d)).strip().splitlines()[-1])
    return kid


def wait(kid: str, poll: int = 30) -> str:
    t0 = time.time()
    while True:
        status = kaggle("kernels", "status", kid, check=False).strip().splitlines()[-1]
        print(f"  {time.time() - t0:6.0f} s  {status}", flush=True)
        if any(k in status.lower() for k in ("complete", "error", "cancel")):
            return status
        time.sleep(poll)


def pull(kid: str) -> None:
    out = ROOT / "kaggle" / "out"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    print(kaggle("kernels", "output", kid, "-p", str(out)).strip().splitlines()[-1])
    (ROOT / "results/raw").mkdir(parents=True, exist_ok=True)
    for f in out.rglob("*"):
        if not f.is_file():
            continue
        if f.name == "triton_load.json":
            shutil.copy(f, ROOT / "results" / f.name)
            print("  pulled results/triton_load.json")
        elif f.suffix in (".json", ".log") and f.name.startswith("triton_"):
            shutil.copy(f, ROOT / "results/raw" / f.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--pull-only", action="store_true")
    a = ap.parse_args()
    kid = f"{a.user}/sla-gateway-triton"
    if not a.pull_only:
        slug = push_source(a.user)
        kid = push_kernel(a.user, slug)
        print("final:", wait(kid))
    pull(kid)


if __name__ == "__main__":
    main()
