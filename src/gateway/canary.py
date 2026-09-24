"""Canary rollout for the accurate tier: 10% -> 50% -> 100%, or roll back.

At each stage the canary gets `weight` of the accurate tier's traffic. After the
stage has lasted `stage_s` seconds AND both versions have at least
`min_requests` in the window, the canary is compared with the stable version:

- error rate must not exceed stable + `max_error_delta`
- p95 must not exceed stable p95 x `max_p95_ratio` + `p95_slack_ms`

Pass: go to the next stage (after the last one, the canary is promoted).
Fail: weight drops to 0 at once and the rollout is marked rolled_back.
A canary that never gets enough traffic within `max_stage_s` also rolls back, and
the gateway aborts the rollout at once if the canary's circuit breaker opens.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .stats import Snapshot

IDLE, RUNNING, PROMOTED, ROLLED_BACK = "idle", "running", "promoted", "rolled_back"


@dataclass
class CanaryConfig:
    stages: tuple[float, ...] = (0.1, 0.5, 1.0)
    stage_s: float = 30.0
    max_stage_s: float = 180.0
    min_requests: int = 50
    max_error_delta: float = 0.02
    max_p95_ratio: float = 1.25
    p95_slack_ms: float = 20.0


@dataclass
class CanaryRollout:
    config: CanaryConfig = field(default_factory=CanaryConfig)
    state: str = IDLE
    stage: int = 0
    stage_started: float = 0.0
    reason: str = ""
    history: list[dict] = field(default_factory=list)

    @property
    def weight(self) -> float:
        return self.config.stages[self.stage] if self.state == RUNNING else 0.0

    def start(self, now: float) -> None:
        self.state, self.stage, self.stage_started, self.reason = RUNNING, 0, now, "started"
        self.history = [{"t": now, "event": "start", "weight": self.weight}]

    def verdict(self, stable: Snapshot, canary: Snapshot) -> tuple[bool, str]:
        c = self.config
        if canary.error_rate > stable.error_rate + c.max_error_delta:
            return False, f"canary error rate {canary.error_rate:.1%} vs stable {stable.error_rate:.1%}"
        if canary.p95_ms is not None and stable.p95_ms is not None:
            limit = stable.p95_ms * c.max_p95_ratio + c.p95_slack_ms
            if canary.p95_ms > limit:
                return False, f"canary p95 {canary.p95_ms:.0f} ms > limit {limit:.0f} ms"
        return True, "healthy"

    def tick(self, now: float, stable: Snapshot, canary: Snapshot) -> str:
        """Advance the rollout. Returns the state after this tick."""
        if self.state != RUNNING:
            return self.state
        c = self.config
        elapsed = now - self.stage_started
        # The last stage sends everything to the canary, so there is no stable
        # traffic left to compare with; judge it on its own error rate instead.
        last = self.stage == len(c.stages) - 1
        enough = canary.count >= c.min_requests and (last or stable.count >= c.min_requests)
        if enough:
            baseline = Snapshot(count=1, errors=0, p50_ms=None, p95_ms=None) if last else stable
            ok, why = self.verdict(baseline, canary)
            if not ok:
                return self._finish(now, ROLLED_BACK, why)
        if elapsed < c.stage_s:
            return self.state
        if not enough:
            if elapsed >= c.max_stage_s:
                return self._finish(now, ROLLED_BACK, "not enough traffic to judge the canary")
            return self.state
        if last:
            return self._finish(now, PROMOTED, "all stages healthy")
        self.stage += 1
        self.stage_started = now
        self.history.append({"t": now, "event": "advance", "weight": self.weight})
        return self.state

    def abort(self, now: float, reason: str) -> str:
        """Roll back right away, e.g. when the canary's circuit breaker has opened."""
        return self._finish(now, ROLLED_BACK, reason) if self.state == RUNNING else self.state

    def _finish(self, now: float, state: str, reason: str) -> str:
        self.state, self.reason = state, reason
        self.history.append({"t": now, "event": state, "weight": 0.0, "reason": reason})
        return state
