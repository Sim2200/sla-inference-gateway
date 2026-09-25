"""Run the benchmark experiments against the Docker Compose stack.

Needs the stack up (`make up`, plus `make canary` for the canary experiments).
Uses only the standard library on the host; the load itself is generated
inside the compose network by the loadgen container, so host networking does
not distort the latency numbers.

    python3 loadtest/experiments.py capacity
    python3 loadtest/experiments.py steady
    python3 loadtest/experiments.py spike
    python3 loadtest/experiments.py canary
    python3 loadtest/experiments.py shadow
    python3 loadtest/experiments.py all

Results go to results/raw/ (per-request CSV + summaries) and results/*.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

GATEWAY = "http://localhost:8080"
RAW = Path("results/raw")
SLA_MS = 300

# Filled in from the capacity experiment (see README / report); override on the command line.
ACCURATE_CAPACITY_RPS = 12.0


def http(method: str, path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(GATEWAY + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def set_mode(mode: str) -> None:
    http("POST", "/admin/mode", {"mode": mode})


def loadgen(profile: str, out: str, warmup_s: float = 0, poll: bool = False, timeout: float = 30) -> dict:
    cmd = ["docker", "compose", "run", "--rm", "-T", "loadgen", "python", "loadtest/loadgen.py",
           "--url", "http://gateway:8080", "--profile", profile, "--out", str(RAW / out),
           "--sla-ms", str(SLA_MS), "--warmup-s", str(warmup_s), "--timeout", str(timeout)]
    if poll:
        cmd.append("--poll-state")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return json.loads((RAW / f"{out}.json").read_text())


def settle(seconds: float = 8, timeout: float = 120) -> None:
    """Wait for queues to drain, then until a probe request comes back fast.

    A laptop has background noise (OS scans, the Docker VM); starting a run while
    something else is hogging the CPU would record that noise as a result.
    """
    time.sleep(seconds)
    probe = next(Path("data/imagenetv2-matched-frequency-format-val/0").glob("*.jpeg")).read_bytes()
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = http("GET", "/state")
        if all(t["in_flight"] == 0 for t in state["tiers"].values()):
            start = time.time()
            req = urllib.request.Request(GATEWAY + "/predict", data=probe, method="POST")
            with urllib.request.urlopen(req, timeout=10):
                pass
            if time.time() - start < 0.15:
                return
        time.sleep(2)
    print("warning: system did not settle", flush=True)


def capacity(modes: tuple[str, ...] = ("always_accurate", "always_fast"), repeats: int = 3) -> dict:
    """Offered load vs p95 for each tier on its own, to find where each saturates.

    Each point is run `repeats` times and the median p95 is reported, with the
    spread kept so the report can show run-to-run noise.
    """
    rates = {"always_accurate": [8, 12, 14, 16, 18, 20, 24], "always_fast": [20, 30, 40, 50, 60, 70]}
    path = Path("results/capacity.json")
    out = json.loads(path.read_text()) if path.exists() else {}
    for mode in modes:
        out[mode] = []
        for rps in rates[mode]:
            runs = []
            for rep in range(repeats):
                settle()
                set_mode(mode)
                runs.append(loadgen(f"{rps}:45", f"capacity_{mode}_{rps}_r{rep}", warmup_s=10, timeout=60)["overall"])
            p95s = sorted(r["p95_ms"] for r in runs)
            median = runs[[r["p95_ms"] for r in runs].index(p95s[len(p95s) // 2])]
            out[mode].append({"rps": rps, **median, "p95_runs": p95s,
                              "within_sla_runs": sorted(r["within_sla"] for r in runs)})
            print(mode, rps, p95s, flush=True)
            path.write_text(json.dumps(out, indent=2))
    return out


def steady(acc_capacity: float) -> dict:
    """Each policy at normal load (60% of the accurate tier's capacity) and at high load (2x)."""
    loads = {"normal": round(0.6 * acc_capacity, 1), "high": round(2.0 * acc_capacity, 1)}
    out = {"loads_rps": loads, "runs": {}}
    for load, rps in loads.items():
        for mode in ("always_accurate", "always_fast", "sla"):
            set_mode(mode)
            s = loadgen(f"{rps}:120", f"steady_{load}_{mode}", warmup_s=20, timeout=60)["overall"]
            out["runs"][f"{load}/{mode}"] = s
            print(load, mode, json.dumps(s), flush=True)
            settle(15)
    Path("results/steady.json").write_text(json.dumps(out, indent=2))
    return out


def spike(acc_capacity: float) -> dict:
    """Normal -> 2x spike -> normal, to watch the controller react and recover."""
    normal, high = round(0.6 * acc_capacity, 1), round(2.0 * acc_capacity, 1)
    profile = f"{normal}:60,{high}:60,{normal}:90"
    out = {"profile": profile, "runs": {}}
    for mode in ("always_accurate", "sla"):
        set_mode(mode)
        s = loadgen(profile, f"spike_{mode}", poll=True, timeout=60)
        out["runs"][mode] = s
        print(mode, json.dumps(s["phases"]), flush=True)
        settle(20)
    Path("results/spike.json").write_text(json.dumps(out, indent=2))
    return out


def recreate_gateway(**env: str) -> None:
    """Restart the gateway with extra environment (fresh state, original config otherwise)."""
    subprocess.run(["docker", "compose", "up", "-d", "--no-deps", "--force-recreate", "--wait", "gateway"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env={**os.environ, **env})
    wait_ready()


def canary(acc_capacity: float) -> dict:
    """Model upgrade under steady load: stable ResNet-50 v1 int8 -> canary v2 int8.

    Run twice: once with the healthy v2 server, once with a v2 server that has an
    injected fault (8% errors, +150 ms). The first should be promoted, the second
    rolled back.
    """
    rps = round(0.6 * acc_capacity, 1)
    out = {}
    for label, name, url in (("healthy", "resnet50-v2-int8", "http://accurate:8000"),
                             ("faulty", "resnet50-v2-int8-faulty", "http://accurate-bad:8000")):
        recreate_gateway(ACCURATE_URL="http://accurate-v1:8000", ACCURATE_NAME="resnet50-v1-int8")
        start_after = 15

        def start() -> None:
            time.sleep(start_after)
            http("POST", "/admin/canary", {"name": name, "url": url})

        threading.Thread(target=start, daemon=True).start()
        s = loadgen(f"{rps}:150", f"canary_{label}", poll=True)
        status = http("GET", "/admin/canary")
        out[label] = {"canary": name, "summary": s, "rollout": status, "started_at_s": start_after}
        print(label, status["state"], status["reason"], flush=True)
    recreate_gateway()
    Path("results/canary.json").write_text(json.dumps(out, indent=2))
    return out


def shadow(acc_capacity: float) -> dict:
    """Mirror 25% of accurate-tier traffic to the fast tier and measure top-1 agreement."""
    rps = round(0.5 * acc_capacity, 1)
    recreate_gateway(SHADOW_FRACTION="0.25")
    s = loadgen(f"{rps}:120", "shadow", warmup_s=10)
    state = http("GET", "/state")
    recreate_gateway()
    agree, disagree = state["shadow"]["agree"], state["shadow"]["disagree"]
    out = {"summary": s["overall"], "agree": agree, "disagree": disagree,
           "agreement": round(agree / max(1, agree + disagree), 4)}
    Path("results/shadow.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out), flush=True)
    return out


def wait_ready(timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            http("GET", "/healthz")
            return
        except OSError:
            time.sleep(1)
    raise SystemExit("gateway did not come back")


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    cap = float(sys.argv[2]) if len(sys.argv) > 2 else ACCURATE_CAPACITY_RPS
    RAW.mkdir(parents=True, exist_ok=True)
    wait_ready()
    steps = {"capacity": lambda: capacity(), "capacity-accurate": lambda: capacity(("always_accurate",)),
             "steady": lambda: steady(cap), "spike": lambda: spike(cap),
             "canary": lambda: canary(cap), "shadow": lambda: shadow(cap)}
    for name in (["capacity", "steady", "spike", "canary", "shadow"] if what == "all" else [what]):
        print(f"== {name}", flush=True)
        steps[name]()


if __name__ == "__main__":
    main()
