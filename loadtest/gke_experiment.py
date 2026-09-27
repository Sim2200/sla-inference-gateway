"""Autoscaling experiment on GKE (cluster from `make gke-up`).

Same traffic profile as the kind experiment but with a bigger spike, so both the
HPA (pods) and the cluster autoscaler (nodes) have to act. Every 5 s the ready
replicas per tier, the HPA's CPU reading and the number of Ready nodes are
recorded. The load generator runs in-cluster; its results are copied out of the
pod when it finishes. Output: results/raw/gke_<name>* and results/gke.json.

    python loadtest/gke_experiment.py [capacity_rps]
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

NS = "sla"
RAW = Path("results/raw")
JOB = Path("deploy/gke/loadgen-job.yaml")


def kubectl(*args: str, stdin: str | None = None, check: bool = True) -> str:
    r = subprocess.run(["kubectl", "-n", NS, *args], input=stdin, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"kubectl {' '.join(args)} failed: {r.stderr}")
    return r.stdout


def snapshot() -> dict:
    deploys = json.loads(kubectl("get", "deploy", "accurate", "fast", "-o", "json"))["items"]
    out = {d["metadata"]["name"]: d["status"].get("readyReplicas", 0) for d in deploys}
    for h in json.loads(kubectl("get", "hpa", "-o", "json"))["items"]:
        metrics = h.get("status", {}).get("currentMetrics") or []
        out[f"{h['metadata']['name']}_cpu_pct"] = next((m["resource"]["current"].get("averageUtilization")
                                                       for m in metrics if m.get("type") == "Resource"), None)
    nodes = json.loads(subprocess.run(["kubectl", "get", "nodes", "-o", "json"], capture_output=True, text=True).stdout)["items"]
    out["nodes_ready"] = sum(1 for n in nodes if any(c["type"] == "Ready" and c["status"] == "True" for c in n["status"]["conditions"]))
    out["nodes_total"] = len(nodes)
    return out


def run(name: str, profile: str) -> dict:
    kubectl("delete", "job", "loadgen", "--ignore-not-found", "--wait=true")
    out_in_pod = f"/tmp/gke_{name}"
    manifest = JOB.read_text().replace("PROFILE", profile).replace("OUT", out_in_pod)
    kubectl("apply", "-f", "-", stdin=manifest)
    timeline, t0 = [], time.time()
    pod = None
    while True:
        timeline.append({"t": round(time.time() - t0, 1), **snapshot()})
        pods = json.loads(kubectl("get", "pods", "-l", "job-name=loadgen", "-o", "json"))["items"]
        pod = pods[0]["metadata"]["name"] if pods else None
        done = pod and subprocess.run(["kubectl", "-n", NS, "exec", pod, "--", "test", "-f", out_in_pod + ".done"],
                                      capture_output=True).returncode == 0
        failed = pods and pods[0]["status"].get("phase") == "Failed"
        if done or failed:
            break
        time.sleep(5)
    if failed:
        print(kubectl("logs", pod, check=False))
        raise SystemExit("load generator failed")
    RAW.mkdir(parents=True, exist_ok=True)
    for suffix in (".json", ".csv", "_state.json"):
        subprocess.run(["kubectl", "-n", NS, "cp", f"{pod}:{out_in_pod}{suffix}", str(RAW / f"gke_{name}{suffix}")], check=True)
    (RAW / f"gke_{name}_replicas.json").write_text(json.dumps(timeline))
    kubectl("delete", "job", "loadgen", "--wait=false")
    summary = json.loads((RAW / f"gke_{name}.json").read_text())
    print(name, json.dumps(summary["phases"]), flush=True)
    return summary


def main() -> None:
    capacity = float(sys.argv[1]) if len(sys.argv) > 1 else 18.0
    normal, high = round(0.6 * capacity, 1), round(4.0 * capacity, 1)
    profile = f"{normal}:60,{high}:360,{normal}:180"
    print(f"profile {profile}")
    kubectl("rollout", "status", "deploy/accurate", "deploy/fast", "deploy/gateway", "--timeout=300s")
    before = snapshot()
    summary = run("hpa_nodes", profile)
    Path("results/gke.json").write_text(json.dumps({"profile": profile, "capacity_rps": capacity, "before": before,
                                                    "after": snapshot(), "phases": summary["phases"]}, indent=2))


if __name__ == "__main__":
    main()
