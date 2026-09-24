"""Open-loop load generator for the gateway.

Requests arrive as a Poisson process at the configured rate, whether or not
earlier requests have finished. Latency is measured from each request's
*scheduled* send time. A closed-loop tool (fixed number of users that wait for
each reply) sends less when the server slows down and so hides queueing delay;
this does not.

Every request is a real ImageNetV2 evaluation image, so the run also measures
top-1 accuracy of whatever tier answered.

Profile syntax: "rps:seconds,rps:seconds,..."  e.g. "8:60,24:60,8:60"

    python loadtest/loadgen.py --url http://gateway:8080 --profile 10:60 --out results/raw/x
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from dataset import split  # noqa: E402


@dataclass
class Result:
    t: float            # scheduled send time, seconds since start
    phase: int
    latency_ms: float
    status: int
    tier: str
    reason: str
    correct: int        # 1 if top-1 matched the label (only meaningful when status == 200)


def parse_profile(text: str) -> list[tuple[float, float]]:
    return [(float(r), float(d)) for r, d in (part.split(":") for part in text.split(","))]


def arrival_times(profile: list[tuple[float, float]], seed: int) -> list[tuple[float, int]]:
    rng = random.Random(seed)
    times, start = [], 0.0
    for phase, (rps, duration) in enumerate(profile):
        t = start
        while rps > 0:
            t += rng.expovariate(rps)
            if t >= start + duration:
                break
            times.append((t, phase))
        start += duration
    return times


def load_images(n: int, seed: int) -> list[tuple[bytes, int]]:
    _, evaluation = split()
    chosen = random.Random(seed).sample(evaluation, n)
    return [(path.read_bytes(), label) for path, label in chosen]


async def run(url: str, profile: list[tuple[float, float]], images: list[tuple[bytes, int]],
              timeout: float, seed: int, poll_state: bool) -> tuple[list[Result], list[dict]]:
    schedule = arrival_times(profile, seed)
    results: list[Result] = []
    states: list[dict] = []
    limits = httpx.Limits(max_connections=2000, max_keepalive_connections=500)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        t0 = time.monotonic()

        async def one(i: int, t: float, phase: int) -> None:
            body, label = images[i % len(images)]
            status, tier, reason, correct = 0, "none", "client_timeout", 0
            try:
                r = await client.post(f"{url}/predict", content=body)
                status = r.status_code
                data = r.json()
                tier, reason = data.get("routed_to", "none"), data.get("reason", "")
                if status == 200:
                    correct = int(data["top5"][0]["index"] == label)
            except httpx.TimeoutException:
                status = 599
            except (httpx.HTTPError, ValueError):
                status, reason = status or 598, "client_error"
            latency = (time.monotonic() - t0 - t) * 1000
            results.append(Result(round(t, 4), phase, round(latency, 2), status, tier, reason, correct))

        async def poller() -> None:
            while True:
                try:
                    s = (await client.get(f"{url}/state", timeout=2)).json()
                    states.append({"t": round(time.monotonic() - t0, 2), "fast_share": s["fast_share"],
                                   "limits": s["limits"], "canary": {k: s["canary"][k] for k in ("state", "weight", "stable", "canary")},
                                   "in_flight": {k: v["in_flight"] for k, v in s["tiers"].items()}})
                except (httpx.HTTPError, KeyError, ValueError):
                    pass
                await asyncio.sleep(1.0)

        poll = asyncio.create_task(poller()) if poll_state else None
        tasks = []
        for i, (t, phase) in enumerate(schedule):
            delay = t - (time.monotonic() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(i, t, phase)))
        await asyncio.gather(*tasks)
        if poll:
            poll.cancel()
    results.sort(key=lambda r: r.t)
    return results, states


def summarize(results: list[Result], sla_ms: float, duration_s: float) -> dict:
    if not results:
        return {"requests": 0}
    lat = np.array([r.latency_ms for r in results])
    ok = np.array([r.status == 200 for r in results])
    ok_lat = lat[ok]
    fast = np.array([r.tier == "fast" for r in results])
    correct = np.array([r.correct for r in results])
    pct = lambda a, q: round(float(np.percentile(a, q)), 1) if len(a) else None  # noqa: E731
    return {
        "requests": len(results),
        "offered_rps": round(len(results) / duration_s, 2),
        "goodput_rps": round(float((ok & (lat <= sla_ms)).sum()) / duration_s, 2),
        "p50_ms": pct(ok_lat, 50), "p95_ms": pct(ok_lat, 95), "p99_ms": pct(ok_lat, 99),
        "within_sla": round(float((ok & (lat <= sla_ms)).mean()), 4),
        "error_rate": round(float((~ok).mean()), 4),
        "shed_rate": round(float(np.mean([r.reason == "overloaded" for r in results])), 4),
        "fast_share": round(float(fast[ok].mean()), 4) if ok.any() else None,
        "top1_accuracy": round(float(correct[ok].mean()), 4) if ok.any() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://gateway:8080")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", required=True, help="output path prefix (writes .csv and .json)")
    parser.add_argument("--sla-ms", type=float, default=300)
    parser.add_argument("--images", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup-s", type=float, default=0, help="exclude the first N seconds from the summary")
    parser.add_argument("--poll-state", action="store_true", help="record gateway /state every second")
    args = parser.parse_args()

    profile = parse_profile(args.profile)
    images = load_images(args.images, args.seed)
    results, states = asyncio.run(run(args.url, profile, images, args.timeout, args.seed, args.poll_state))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)
    measured = [r for r in results if r.t >= args.warmup_s]
    total = sum(d for _, d in profile) - args.warmup_s
    summary = {"profile": args.profile, "warmup_s": args.warmup_s, "sla_ms": args.sla_ms,
               "overall": summarize(measured, args.sla_ms, total), "phases": []}
    start = 0.0
    for phase, (rps, duration) in enumerate(profile):
        rows = [r for r in measured if r.phase == phase]
        eff = duration - max(0.0, min(duration, args.warmup_s - start))
        summary["phases"].append({"rps": rps, "duration_s": duration, **summarize(rows, args.sla_ms, eff)})
        start += duration
    if states:
        out.with_name(out.name + "_state.json").write_text(json.dumps(states))
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["overall"]))


if __name__ == "__main__":
    main()
