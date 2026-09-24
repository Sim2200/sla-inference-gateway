"""SLA controller: decides what share of traffic goes to the fast tier.

Two mechanisms work together:

1. Feedback (every tick, e.g. 1 s). Look at the accurate tier's p95 over the
   last window. If it is above `high_water` x SLA, or its error rate is too high,
   move `step_up` of traffic to the fast tier at once. Only after `cooldown_s`
   of calm (p95 below `low_water` x SLA) give back `step_down` per tick.
   Fast to protect, slow to recover: the gap between the two thresholds and the
   cooldown stop the split from flapping.

2. Admission (every request). If the accurate tier already has `max_inflight`
   requests outstanding, the request spills to the fast tier immediately,
   without waiting for the next tick. This catches bursts shorter than a tick.
"""

from __future__ import annotations

from dataclasses import dataclass

from .stats import Snapshot


@dataclass
class ControllerConfig:
    sla_ms: float = 300.0
    high_water: float = 0.8
    low_water: float = 0.5
    step_up: float = 0.25
    step_down: float = 0.05
    cooldown_s: float = 5.0
    raise_hold_s: float = 2.0  # minimum gap between two raises, so window lag does not cause overshoot
    window_s: float = 5.0
    min_samples: int = 10
    max_error_rate: float = 0.05


class SlaController:
    def __init__(self, config: ControllerConfig) -> None:
        self.config = config
        self.fast_share = 0.0
        self.last_reason = "within_sla"
        self._last_raise = float("-inf")

    def tick(self, now: float, accurate: Snapshot) -> float:
        c = self.config
        enough = accurate.count >= c.min_samples
        over_latency = enough and accurate.p95_ms is not None and accurate.p95_ms > c.high_water * c.sla_ms
        over_errors = enough and accurate.error_rate > c.max_error_rate
        if over_latency or over_errors:
            if now - self._last_raise >= c.raise_hold_s:
                self.fast_share = min(1.0, self.fast_share + c.step_up)
                self._last_raise = now
            self.last_reason = "p95_above_target" if over_latency else "error_rate_above_target"
        elif now - self._last_raise >= c.cooldown_s:
            # With too few samples (e.g. all traffic already on the fast tier) there is
            # no evidence of trouble, so recover slowly and let fresh samples decide.
            calm = not enough or accurate.p95_ms is None or accurate.p95_ms < c.low_water * c.sla_ms
            if calm and self.fast_share > 0:
                self.fast_share = max(0.0, round(self.fast_share - c.step_down, 4))
                self.last_reason = "recovering"
            elif self.fast_share == 0:
                self.last_reason = "within_sla"
        return self.fast_share


@dataclass
class LimitConfig:
    initial: int = 4
    minimum: int = 1
    maximum: int = 64
    backoff: float = 0.75
    hold_s: float = 3.0  # after a cut, wait this long for the window to reflect it before cutting again


class AdaptiveLimit:
    """How many requests a tier may have outstanding at once (AIMD).

    Latency above the high-water mark cuts the limit by `backoff`. When the limit
    was actually reached since the last tick and latency is comfortably low, it
    grows by one. So when the autoscaler adds replicas, latency drops and the
    limit climbs to use them, without the gateway knowing the replica count.
    """

    def __init__(self, config: LimitConfig, sla: ControllerConfig) -> None:
        self.config, self.sla = config, sla
        self.limit = config.initial
        self._hit_limit = False
        self._last_cut = float("-inf")

    def note_saturated(self) -> None:
        self._hit_limit = True

    def tick(self, now: float, snapshot: Snapshot) -> int:
        c, s = self.config, self.sla
        enough = snapshot.count >= s.min_samples and snapshot.p95_ms is not None
        if enough and snapshot.p95_ms > s.high_water * s.sla_ms:
            if now - self._last_cut >= c.hold_s:
                self.limit = max(c.minimum, int(self.limit * c.backoff))
                self._last_cut = now
        elif self._hit_limit and (not enough or snapshot.p95_ms < s.low_water * s.sla_ms):
            self.limit = min(c.maximum, self.limit + 1)
        self._hit_limit = False
        return self.limit
