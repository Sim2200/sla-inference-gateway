"""Per-request routing decision. Pure function of the current state, so it is easy to test."""

from __future__ import annotations

from dataclasses import dataclass

ACCURATE, FAST = "accurate", "fast"

# Policy modes. The two "always" modes are the naive baselines used in the benchmarks:
# a plain proxy to one tier, with no admission control and no fallback.
SLA, ALWAYS_ACCURATE, ALWAYS_FAST = "sla", "always_accurate", "always_fast"
MODES = (SLA, ALWAYS_ACCURATE, ALWAYS_FAST)


@dataclass(frozen=True)
class TierView:
    in_flight: int
    max_in_flight: int
    breaker_allows: bool

    @property
    def saturated(self) -> bool:
        return self.in_flight >= self.max_in_flight

    @property
    def available(self) -> bool:
        return self.breaker_allows and not self.saturated


@dataclass(frozen=True)
class Decision:
    tier: str | None  # None means shed the request (503)
    reason: str


def decide(mode: str, fast_share: float, draw: float, accurate: TierView, fast: TierView) -> Decision:
    """Pick a tier for one request.

    `draw` is a uniform random number in [0, 1); passing it in keeps this deterministic in tests.
    """
    if mode == ALWAYS_ACCURATE:
        return Decision(ACCURATE, "baseline_always_accurate")
    if mode == ALWAYS_FAST:
        return Decision(FAST, "baseline_always_fast")

    if not accurate.breaker_allows:
        preferred, reason = FAST, "accurate_breaker_open"
    elif accurate.saturated:
        preferred, reason = FAST, "accurate_saturated"
    elif draw < fast_share:
        preferred, reason = FAST, "sla_pressure"
    else:
        preferred, reason = ACCURATE, "within_sla"

    if preferred == ACCURATE:
        return Decision(ACCURATE, reason)
    if fast.available:
        return Decision(FAST, reason)
    # The fast tier cannot take it either. Use the accurate tier if it has room,
    # otherwise shed: a quick 503 is better than a request that blows the SLA
    # and makes every request behind it slower too.
    if accurate.available:
        return Decision(ACCURATE, "fast_unavailable")
    return Decision(None, "overloaded")
