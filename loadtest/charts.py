"""Render the result JSON/CSV files into report figures (report/figures/*.png).

Colors: one fixed hue per policy across every figure (validated categorical
slots 1-3), neutral ink for all text, hairline grids, and no dual axes: when
two measures share a time axis they get their own panels.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS, RAW, FIG = Path("results"), Path("results/raw"), Path("report/figures")

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COLORS = {"sla": BLUE, "always_accurate": ORANGE, "always_fast": AQUA,
          "hpa": BLUE, "fixed": ORANGE}
LABELS = {"sla": "SLA gateway", "always_accurate": "Always accurate", "always_fast": "Always fast",
          "hpa": "SLA gateway + HPA", "fixed": "SLA gateway, 1 replica per tier"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SLA_MS = 300

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "legend.labelcolor": INK2, "lines.linewidth": 2,
})


def load(name: str):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def rows(name: str) -> list[dict]:
    path = RAW / f"{name}.csv"
    if not path.exists():
        return []
    with path.open() as f:
        return [{"t": float(r["t"]), "lat": float(r["latency_ms"]), "status": int(r["status"]),
                 "tier": r["tier"], "reason": r["reason"]} for r in csv.DictReader(f)]


def binned(data: list[dict], width: float, fn) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for r in data:
        groups[int(r["t"] // width)].append(r)
    keys = sorted(groups)
    return np.array([(k + 0.5) * width for k in keys]), np.array([fn(groups[k]) for k in keys])


def p95_ok(group: list[dict]) -> float:
    lat = [r["lat"] for r in group if r["status"] == 200]
    return float(np.percentile(lat, 95)) if lat else np.nan


def sla_line(ax) -> None:
    ax.axhline(SLA_MS, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("SLA 300 ms", xy=(1, SLA_MS), xycoords=("axes fraction", "data"), xytext=(-4, 4),
                textcoords="offset points", ha="right", va="bottom", fontsize=8, color=INK2)


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


# ---- figures -----------------------------------------------------------------

def models() -> None:
    data = load("models.json")
    if not data:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, m in data.items():
        if "latency_ms_p50" not in m or "top1" not in m:
            continue
        int8 = m["precision"] == "int8"
        ax.scatter(m["latency_ms_p50"], m["top1"] * 100, s=46, color=BLUE if not int8 else ORANGE,
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(name.replace("_", " "), (m["latency_ms_p50"], m["top1"] * 100), xytext=(6, 3),
                    textcoords="offset points", fontsize=8, color=INK2)
    ax.scatter([], [], color=BLUE, label="fp32")
    ax.scatter([], [], color=ORANGE, label="int8 (static quantization)")
    ax.set_xscale("log")
    ax.set_xlabel("Inference latency, batch 1, 2 CPU threads (ms, log scale)")
    ax.set_ylabel("ImageNetV2 top-1 accuracy (%)")
    ax.set_title("Accuracy vs latency of the candidate models", loc="left")
    ax.legend(loc="lower right")
    save(fig, "models.png")


def capacity() -> None:
    data = load("capacity.json")
    if not data:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for mode, points in data.items():
        x = [p["rps"] for p in points]
        y = [p["p95_ms"] or np.nan for p in points]
        ax.plot(x, y, marker="o", markersize=5, color=COLORS[mode], label=LABELS[mode])
    sla_line(ax)
    ax.set_yscale("log")
    ax.set_xlabel("Offered load (requests/s, Poisson arrivals)")
    ax.set_ylabel("p95 latency (ms, log scale)")
    ax.set_title("Each tier on its own: where latency breaks", loc="left")
    ax.legend(loc="upper left")
    save(fig, "capacity.png")


def steady() -> None:
    data = load("steady.json")
    if not data:
        return
    modes = ["always_accurate", "always_fast", "sla"]
    loads = list(data["loads_rps"])
    metrics = [("p95_ms", "p95 latency (ms, log)", True),
               ("within_sla", "Requests answered within SLA (%)", False),
               ("top1_accuracy", "Top-1 accuracy of answers (%)", False)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    width = 0.26
    x = np.arange(len(loads))
    for ax, (key, title, log) in zip(axes, metrics):
        for i, mode in enumerate(modes):
            vals = []
            for level in loads:
                v = data["runs"][f"{level}/{mode}"].get(key)
                vals.append(np.nan if v is None else (v if key == "p95_ms" else v * 100))
            bars = ax.bar(x + (i - 1) * width, vals, width - 0.03, color=COLORS[mode], label=LABELS[mode])
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    label = f"{v:,.0f}" if key == "p95_ms" else f"{v:.0f}"
                    ax.annotate(label, (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 2),
                                textcoords="offset points", ha="center", fontsize=7, color=INK2)
        ax.set_xticks(x, [f"{level}\n{data['loads_rps'][level]} req/s" for level in loads])
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
        if log:
            ax.set_yscale("log")
            sla_line(ax)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Steady load: normal (60% of accurate-tier capacity) and high (2x)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "steady.png")


def spike() -> None:
    data = load("spike.json")
    if not data:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, height_ratios=[3, 2])
    for mode in ("always_accurate", "sla"):
        r = rows(f"spike_{mode}")
        t, p95 = binned(r, 5, p95_ok)
        ax1.plot(t, p95, color=COLORS[mode], label=LABELS[mode])
        t, share = binned(r, 5, lambda g: 100 * np.mean([x["tier"] == "fast" for x in g if x["status"] == 200] or [0]))
        ax2.plot(t, share, color=COLORS[mode], label=LABELS[mode])
    for ax in (ax1, ax2):
        ax.axvspan(60, 120, color=GRID, alpha=0.5, linewidth=0)
    ax1.annotate("2x load spike", (90, 1), xycoords=("data", "axes fraction"), xytext=(0, -12),
                 textcoords="offset points", ha="center", fontsize=8, color=INK2)
    sla_line(ax1)
    ax1.set_yscale("log")
    ax1.set_ylabel("p95 latency per 5 s (ms, log)")
    ax1.set_title("Load spike: latency and where traffic went", loc="left")
    ax1.legend(loc="upper left")
    ax2.set_ylabel("Answered by fast tier (%)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim(-3, 103)
    save(fig, "spike.png")


def canary() -> None:
    data = load("canary.json")
    if not data:
        return
    names = list(data)
    fig, axes = plt.subplots(2, len(names), figsize=(11, 5.2), sharex=True, height_ratios=[2, 3])
    for col, name in enumerate(names):
        run = data[name]
        state_path = RAW / f"canary_{name}_state.json"
        states = json.loads(state_path.read_text()) if state_path.exists() else []
        ax_w, ax_e = axes[0][col], axes[1][col]
        ax_w.step([s["t"] for s in states], [100 * s["canary"]["weight"] for s in states], where="post", color=BLUE)
        ax_w.set_ylim(-5, 105)
        ax_w.set_title(f"{name} v2 canary: {run['rollout']['state'].replace('_', ' ')}", loc="left", fontsize=10)
        ax_w.set_ylabel("Canary weight (%)" if col == 0 else "")
        r = rows(f"canary_{name}")
        t, err = binned(r, 5, lambda g: 100 * np.mean([x["status"] != 200 for x in g]))
        ax_e.plot(t, err, color=ORANGE, label="client-visible errors (%)")
        ax_e.set_ylabel("Errors seen by clients (%)" if col == 0 else "")
        ax_e.set_xlabel("Time (s)")
        ax_e.set_ylim(-1, max(10, np.nanmax(err) + 2 if len(err) else 10))
        reason = run["rollout"].get("reason", "")
        ax_w.annotate(reason, (0.99, 0.08), xycoords="axes fraction", ha="right", fontsize=7.5, color=INK2)
    fig.suptitle("Canary rollouts under steady load (10% -> 50% -> 100%)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "canary.png")


def k8s() -> None:
    info = load("k8s_hpa.json")
    if not info:
        return
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.6), sharex=True, height_ratios=[3, 2, 2])
    for run in ("fixed", "hpa"):
        r = rows(f"k8s_{run}")
        t, p95 = binned(r, 10, p95_ok)
        axes[0].plot(t, p95, color=COLORS[run], label=LABELS[run])
        t, share = binned(r, 10, lambda g: 100 * np.mean([x["tier"] == "fast" for x in g if x["status"] == 200] or [0]))
        axes[1].plot(t, share, color=COLORS[run], label=LABELS[run])
    rep_path = RAW / "k8s_hpa_replicas.json"
    if rep_path.exists():
        reps = json.loads(rep_path.read_text())
        axes[2].step([x["t"] for x in reps], [x.get("accurate", 0) for x in reps], where="post", color=BLUE,
                     label="accurate tier replicas")
        axes[2].step([x["t"] for x in reps], [x.get("fast", 0) for x in reps], where="post", color=AQUA,
                     label="fast tier replicas")
        axes[2].legend(loc="upper right", fontsize=8)
    sla_line(axes[0])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("p95 per 10 s (ms, log)")
    axes[0].set_title("Kubernetes: the gateway with and without the autoscaler", loc="left")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].set_ylabel("Answered by fast tier (%)")
    axes[1].set_ylim(-3, 103)
    axes[2].set_ylabel("Ready replicas (HPA run)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_yticks([0, 1, 2, 3, 4])
    for ax in axes:
        ax.axvspan(60, 300, color=GRID, alpha=0.5, linewidth=0)
    save(fig, "k8s_hpa.png")


if __name__ == "__main__":
    for fn in (models, capacity, steady, spike, canary, k8s):
        fn()
