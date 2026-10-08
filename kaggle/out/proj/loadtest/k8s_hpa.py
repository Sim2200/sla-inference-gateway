"""Autoscaling experiment on the kind cluster (run `make k8s-up` first).

The same traffic profile is run twice against the SLA gateway:
  1. fixed: HPA removed, one replica per tier
  2. hpa:   HPA on (accurate 1-4 replicas, fast 1-2)

While the in-cluster load generator runs, replica counts and HPA CPU readings
are recorded every 5 seconds. Everything lands in results/raw/k8s_*.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

NS = "sla"
RAW = Path("results/raw")
JOB = Path("deploy/k8s/loadgen-job.yaml")


def kubectl(*args: str, stdin: str | None = None, check: bool = True) -> str:
    r = subprocess.run(["kubectl", "-n", NS, *args], input=stdin, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"kubectl {' '.join(args)} failed: {r.stderr}")
    return r.stdout


def replicas() -> dict:
    deploys = json.loads(kubectl("get", "deploy", "accurate", "fast", "-o", "json"))["items"]
    out = {d["metadata"]["name"]: d["status"].get("readyReplicas", 0) for d in deploys}
    hpas = json.loads(kubectl("get", "hpa", "-o", "json", check=False) or '{"items": []}')["items"]
    for h in hpas:
        metrics = h.get("status", {}).get("currentMetrics") or []
        cpu = next((m["resource"]["current"].get("averageUtilization") for m in metrics
                    if m.get("type") == "Resource"), None)
        out[f"{h['metadata']['name']}_cpu_pct"] = cpu
    return out


def scale_to_one() -> None:
    kubectl("delete", "hpa", "--all", check=False)
    kubectl("scale", "deploy/accurate", "deploy/fast", "--replicas=1")
    kubectl("rollout", "status", "deploy/accurate", "deploy/fast", "--timeout=180s")


def run(name: str, profile: str) -> None:
    kubectl("delete", "job", "loadgen", "--ignore-not-found", "--wait=true")
    manifest = JOB.read_text().replace("PROFILE", profile).replace("OUT", f"results/raw/k8s_{name}")
    kubectl("apply", "-f", "-", stdin=manifest)
    timeline, t0 = [], time.time()
    while True:
        timeline.append({"t": round(time.time() - t0, 1), **replicas()})
        status = json.loads(kubectl("get", "job", "loadgen", "-o", "json"))["status"]
        if status.get("succeeded") or status.get("failed"):
            break
        time.sleep(5)
    if status.get("failed"):
        print(kubectl("logs", "job/loadgen", check=False))
        raise SystemExit("load generator failed")
    (RAW / f"k8s_{name}_replicas.json").write_text(json.dumps(timeline))
    summary = json.loads((RAW / f"k8s_{name}.json").read_text())
    print(name, json.dumps(summary["phases"]), flush=True)


def main() -> None:
    capacity = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
    normal, high = round(0.6 * capacity, 1), round(2.5 * capacity, 1)
    profile = f"{normal}:60,{high}:240,{normal}:120"
    RAW.mkdir(parents=True, exist_ok=True)

    scale_to_one()
    run("fixed", profile)
    time.sleep(30)

    kubectl("apply", "-f", "deploy/k8s/hpa.yaml")
    time.sleep(30)  # let metrics-server report before the load starts
    run("hpa", profile)
    Path("results/k8s_hpa.json").write_text(json.dumps({"profile": profile, "capacity_rps": capacity}, indent=2))


if __name__ == "__main__":
    main()
