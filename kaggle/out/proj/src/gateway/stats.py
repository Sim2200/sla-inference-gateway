"""Time-windowed request statistics (latency percentiles and error rate)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import math


@dataclass(frozen=True)
class Snapshot:
    count: int
    errors: int
    p50_ms: float | None
    p95_ms: float | None

    @property
    def error_rate(self) -> float:
        return self.errors / self.count if self.count else 0.0


class RollingWindow:
    """Keeps (time, latency, ok) samples from the last `window_s` seconds.

    Percentiles use successful requests only: a fast failure should not make a
    backend look healthy. Errors are counted separately.
    """

    def __init__(self, window_s: float = 10.0, max_samples: int = 20_000) -> None:
        self.window_s = window_s
        self._samples: deque[tuple[float, float, bool]] = deque(maxlen=max_samples)

    def add(self, now: float, latency_ms: float, ok: bool) -> None:
        self._samples.append((now, latency_ms, ok))

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def snapshot(self, now: float) -> Snapshot:
        self._prune(now)
        latencies = sorted(lat for _, lat, ok in self._samples if ok)
        errors = sum(1 for _, _, ok in self._samples if not ok)
        return Snapshot(
            count=len(self._samples),
            errors=errors,
            p50_ms=percentile(latencies, 50),
            p95_ms=percentile(latencies, 95),
        )

    def clear(self) -> None:
        self._samples.clear()


def percentile(sorted_values: list[float], q: float) -> float | None:
    """Nearest-rank percentile of an already sorted list (None if empty)."""
    if not sorted_values:
        return None
    rank = max(1, math.ceil(q / 100 * len(sorted_values)))
    return sorted_values[rank - 1]
