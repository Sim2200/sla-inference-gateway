"""Unit tests for the gateway's decision logic (no network)."""

import pytest

from gateway.breaker import CLOSED, HALF_OPEN, OPEN, CircuitBreaker
from gateway.canary import PROMOTED, ROLLED_BACK, RUNNING, CanaryConfig, CanaryRollout
from gateway.controller import AdaptiveLimit, ControllerConfig, LimitConfig, SlaController
from gateway.routing import ACCURATE, ALWAYS_ACCURATE, ALWAYS_FAST, FAST, SLA, TierView, decide
from gateway.stats import RollingWindow, Snapshot, percentile


def snap(p95=None, count=100, errors=0):
    return Snapshot(count=count, errors=errors, p50_ms=p95, p95_ms=p95)


# ---- stats -------------------------------------------------------------------

def test_percentile_nearest_rank():
    values = sorted(range(1, 101))
    assert percentile(values, 95) == 95
    assert percentile(values, 50) == 50
    assert percentile([7.0], 95) == 7.0
    assert percentile([], 95) is None


def test_window_drops_old_samples_and_ignores_errors_for_latency():
    w = RollingWindow(window_s=10)
    w.add(0.0, 1000.0, True)            # will age out
    for i in range(10):
        w.add(5.0 + i * 0.1, 50.0, True)
    w.add(6.0, 5.0, False)              # a fast failure must not lower the p95
    s = w.snapshot(now=12.0)
    assert s.count == 11 and s.errors == 1
    assert s.p95_ms == 50.0
    assert s.error_rate == pytest.approx(1 / 11)


# ---- circuit breaker ---------------------------------------------------------

def test_breaker_opens_after_consecutive_failures_and_recovers_via_probe():
    b = CircuitBreaker(failure_threshold=3, open_s=5)
    for t in range(3):
        assert b.allow(t)
        b.record(t, ok=False)
    assert b.state == OPEN and not b.allow(4) and not b.peek(4)
    assert b.peek(8.0)                   # open period over
    assert b.allow(8.0) and b.state == HALF_OPEN
    assert not b.allow(8.1)              # only one probe at a time
    b.record(8.2, ok=True)
    assert b.state == CLOSED and b.allow(8.3)


def test_breaker_failed_probe_reopens():
    b = CircuitBreaker(failure_threshold=1, open_s=5)
    b.record(0, ok=False)
    assert b.allow(6) and b.state == HALF_OPEN
    b.record(6, ok=False)
    assert b.state == OPEN and not b.allow(7)


def test_breaker_success_resets_failure_count():
    b = CircuitBreaker(failure_threshold=3)
    b.record(0, False); b.record(1, False); b.record(2, True); b.record(3, False); b.record(4, False)
    assert b.state == CLOSED


# ---- SLA controller ----------------------------------------------------------

CFG = ControllerConfig(sla_ms=300, high_water=0.8, low_water=0.5, step_up=0.25, step_down=0.05,
                       cooldown_s=5, raise_hold_s=2, min_samples=10)


def test_controller_raises_fast_share_when_p95_over_high_water():
    c = SlaController(CFG)
    assert c.tick(0, snap(p95=200)) == 0          # 200 < 240: fine
    assert c.tick(1, snap(p95=260)) == 0.25       # 260 > 240: shed load
    assert c.tick(2, snap(p95=260)) == 0.25       # held: window has not caught up yet
    assert c.tick(3, snap(p95=260)) == 0.5
    assert c.last_reason == "p95_above_target"


def test_controller_hysteresis_band_and_cooldown():
    c = SlaController(CFG)
    c.tick(0, snap(p95=400))
    assert c.fast_share == 0.25
    assert c.tick(3, snap(p95=100)) == 0.25        # calm, but still inside the cooldown
    assert c.tick(6, snap(p95=200)) == 0.25        # between low (150) and high (240): hold
    assert c.tick(7, snap(p95=100)) == 0.2         # calm and cooled down: give back slowly
    assert c.last_reason == "recovering"


def test_controller_error_rate_triggers_and_small_samples_are_ignored():
    c = SlaController(CFG)
    assert c.tick(0, snap(p95=1000, count=5)) == 0     # too few samples to act on
    assert c.tick(1, snap(p95=100, count=100, errors=10)) == 0.25
    assert c.last_reason == "error_rate_above_target"


def test_controller_recovers_when_accurate_gets_no_traffic():
    c = SlaController(CFG)
    c.fast_share = 1.0
    assert c.tick(100, snap(p95=None, count=0)) == 0.95


def test_adaptive_limit_cuts_on_high_latency_and_grows_only_when_used():
    lim = AdaptiveLimit(LimitConfig(initial=8, minimum=1, maximum=10, backoff=0.75, hold_s=3), CFG)
    assert lim.tick(0, snap(p95=100)) == 8             # calm but never saturated: no growth
    lim.note_saturated()
    assert lim.tick(1, snap(p95=100)) == 9
    assert lim.tick(2, snap(p95=500)) == 6
    assert lim.tick(3, snap(p95=500)) == 6             # hold after a cut
    assert lim.tick(5, snap(p95=500)) == 4
    for t in range(6, 20):
        lim.note_saturated()
        lim.tick(t, snap(p95=50))
    assert lim.limit == 10                             # capped at maximum


# ---- routing -----------------------------------------------------------------

def view(in_flight=0, limit=4, breaker=True):
    return TierView(in_flight=in_flight, max_in_flight=limit, breaker_allows=breaker)


def test_baselines_ignore_load():
    busy = view(in_flight=100)
    assert decide(ALWAYS_ACCURATE, 1.0, 0.0, busy, busy).tier == ACCURATE
    assert decide(ALWAYS_FAST, 0.0, 0.9, busy, busy).tier == FAST


@pytest.mark.parametrize("acc, fast_share, draw, tier, reason", [
    (view(), 0.0, 0.5, ACCURATE, "within_sla"),
    (view(), 0.3, 0.2, FAST, "sla_pressure"),
    (view(), 0.3, 0.4, ACCURATE, "within_sla"),
    (view(in_flight=4), 0.0, 0.9, FAST, "accurate_saturated"),
    (view(breaker=False), 0.0, 0.9, FAST, "accurate_breaker_open"),
])
def test_sla_routing(acc, fast_share, draw, tier, reason):
    d = decide(SLA, fast_share, draw, acc, view(limit=8))
    assert (d.tier, d.reason) == (tier, reason)


def test_sla_routing_falls_back_to_accurate_then_sheds():
    fast_full = view(in_flight=8, limit=8)
    assert decide(SLA, 1.0, 0.1, view(), fast_full).tier == ACCURATE
    shed = decide(SLA, 0.0, 0.9, view(in_flight=4), fast_full)
    assert shed.tier is None and shed.reason == "overloaded"


# ---- canary ------------------------------------------------------------------

CANARY = CanaryConfig(stages=(0.1, 0.5, 1.0), stage_s=30, max_stage_s=120, min_requests=50,
                      max_error_delta=0.02, max_p95_ratio=1.25, p95_slack_ms=20)


def test_canary_healthy_rollout_promotes_through_all_stages():
    r = CanaryRollout(CANARY)
    r.start(0)
    assert r.weight == 0.1
    healthy = snap(p95=100)
    assert r.tick(10, healthy, healthy) == RUNNING and r.weight == 0.1   # stage not over
    r.tick(30, healthy, healthy)
    assert r.weight == 0.5
    r.tick(60, healthy, healthy)
    assert r.weight == 1.0
    assert r.tick(90, snap(count=0), healthy) == PROMOTED              # last stage: no stable traffic left
    assert [h["event"] for h in r.history] == ["start", "advance", "advance", "promoted"]


def test_canary_rolls_back_immediately_on_errors():
    r = CanaryRollout(CANARY)
    r.start(0)
    assert r.tick(5, snap(p95=100, errors=0), snap(p95=100, errors=10)) == ROLLED_BACK
    assert r.weight == 0.0 and "error rate" in r.reason


def test_canary_rolls_back_on_latency_regression():
    r = CanaryRollout(CANARY)
    r.start(0)
    # limit = 100 * 1.25 + 20 = 145 ms
    assert r.tick(31, snap(p95=100), snap(p95=140)) == RUNNING
    assert r.tick(40, snap(p95=100), snap(p95=150)) == ROLLED_BACK


def test_canary_without_enough_traffic_eventually_rolls_back():
    r = CanaryRollout(CANARY)
    r.start(0)
    assert r.tick(60, snap(p95=100), snap(p95=100, count=3)) == RUNNING
    assert r.tick(121, snap(p95=100), snap(p95=100, count=3)) == ROLLED_BACK
