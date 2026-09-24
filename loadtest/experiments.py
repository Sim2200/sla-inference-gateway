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


def settle(seconds: float = 8) -> None:
    """Let queues drain between runs."""
    time.sleep(seconds)


def capacity() -> dict:
    """Offered load vs p95 for each tier on its own, to find where each saturates."""
    out = {}
    for mode, rates in (("always_accurate", [4, 6, 8, 10, 12, 14, 16]),
                        ("always_fast", [8, 12, 16, 20, 24, 28, 32, 36])):
        out[mode] = []
        for rps in rates:
            set_mode(mode)
            s = loadgen(f"{rps}:45", f"capacity_{mode}_{rps}", warmup_s=10, timeout=60)["overall"]
            out[mode].append({"rps": rps, **s})
            print(mode, rps, s["p95_ms"], s["within_sla"], flush=True)
            settle()
    Path("results/capacity.json").write_text(json.dumps(out, indent=2))
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


def canary(acc_capacity: float) -> dict:
    """Roll out a healthy v2 and a faulty v2 under steady load; record what the controller did."""
    rps = round(0.6 * acc_capacity, 1)
    out = {}
    for name, url in (("resnet50-v2-fp32", "http://accurate-v2:8000"),
                      ("resnet50-v2-faulty", "http://accurate-bad:8000")):
        set_mode("sla")
        start_after = 15

        def start() -> None:
            time.sleep(start_after)
            http("POST", "/admin/canary", {"name": name, "url": url})

        threading.Thread(target=start, daemon=True).start()
        s = loadgen(f"{rps}:150", f"canary_{name}", poll=True)
        status = http("GET", "/admin/canary")
        out[name] = {"summary": s, "rollout": status, "started_at_s": start_after}
        print(name, status["state"], status["reason"], flush=True)
        # Put the original stable version back for the next run.
        subprocess.run(["docker", "compose", "restart", "gateway"], check=True, stdout=subprocess.DEVNULL)
        wait_ready()
    Path("results/canary.json").write_text(json.dumps(out, indent=2))
    return out


def shadow(acc_capacity: float) -> dict:
    """Mirror 25% of accurate-tier traffic to the fast tier and measure top-1 agreement."""
    rps = round(0.5 * acc_capacity, 1)
    env = {"SHADOW_FRACTION": "0.25"}
    subprocess.run(["docker", "compose", "up", "-d", "--no-deps", "--force-recreate", "gateway"],
                   check=True, stdout=subprocess.DEVNULL, env={**os.environ, **env})
    wait_ready()
    set_mode("sla")
    s = loadgen(f"{rps}:120", "shadow", warmup_s=10)
    state = http("GET", "/state")
    subprocess.run(["docker", "compose", "up", "-d", "--no-deps", "--force-recreate", "gateway"],
                   check=True, stdout=subprocess.DEVNULL)
    wait_ready()
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
    steps = {"capacity": lambda: capacity(), "steady": lambda: steady(cap), "spike": lambda: spike(cap),
             "canary": lambda: canary(cap), "shadow": lambda: shadow(cap)}
    for name in (steps if what == "all" else [what]):
        print(f"== {name}", flush=True)
        steps[name]()


if __name__ == "__main__":
    main()
