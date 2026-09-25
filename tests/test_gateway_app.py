"""End-to-end gateway tests with fake backends (httpx.MockTransport)."""

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.canary import CanaryConfig
from gateway.config import GatewayConfig, Target, TierConfig
from gateway.controller import ControllerConfig, LimitConfig


def prediction(tier, index=1):
    return {"tier": tier, "version": "v1", "model": tier, "top5": [{"index": index, "label": "x", "score": 0.9}]}


class FakeBackends:
    """Routes by host name; each backend can be told to fail."""

    def __init__(self):
        self.calls = {"accurate": 0, "fast": 0, "canary": 0}
        self.failing = set()
        self.last_headers = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.last_headers = dict(request.headers)
        host = request.url.host
        self.calls[host] += 1
        if host in self.failing:
            return httpx.Response(500, json={"detail": "boom"})
        tier = "accurate" if host in ("accurate", "canary") else "fast"
        return httpx.Response(200, json=prediction(tier))


def make(mode="sla", canary=None, shadow=0.0):
    cfg = GatewayConfig(
        accurate=TierConfig(Target("acc-v1", "http://accurate"), LimitConfig(initial=4)),
        fast=TierConfig(Target("fast-v1", "http://fast"), LimitConfig(initial=8)),
        mode=mode, controller=ControllerConfig(min_samples=5),
        breaker_failures=3, breaker_open_s=60, canary=canary or CanaryConfig(), shadow_fraction=shadow,
    )
    fake = FakeBackends()
    app = create_app(cfg, transport=httpx.MockTransport(fake), tick_s=0)
    return app, fake


def test_normal_request_goes_to_accurate_tier():
    app, fake = make()
    with TestClient(app) as c:
        r = c.post("/predict", content=b"img")
    assert r.status_code == 200
    body = r.json()
    assert body["routed_to"] == "accurate" and body["reason"] == "within_sla"
    assert r.headers["X-Routed-To"] == "accurate"
    assert fake.calls == {"accurate": 1, "fast": 0, "canary": 0}


def test_empty_body_is_rejected():
    app, _ = make()
    with TestClient(app) as c:
        assert c.post("/predict", content=b"").status_code == 400


def test_accurate_failure_falls_back_and_opens_breaker():
    app, fake = make()
    fake.failing.add("accurate")
    with TestClient(app) as c:
        reasons = [c.post("/predict", content=b"img").json()["reason"] for _ in range(5)]
        state = c.get("/state").json()
    # first three hit the accurate tier and fall back; then the breaker is open
    assert reasons[:3] == ["fallback_after_error"] * 3
    assert reasons[3:] == ["accurate_breaker_open"] * 2
    assert fake.calls["accurate"] == 3
    assert state["backends"]["acc-v1"]["breaker"] == "open"


def test_baseline_mode_does_not_fall_back():
    app, fake = make(mode="always_accurate")
    fake.failing.add("accurate")
    with TestClient(app) as c:
        r = c.post("/predict", content=b"img")
    assert r.status_code == 500 and fake.calls["fast"] == 0


def test_mode_switch_and_validation():
    app, fake = make()
    with TestClient(app) as c:
        assert c.post("/admin/mode", json={"mode": "always_fast"}).json() == {"mode": "always_fast"}
        assert c.post("/predict", content=b"img").json()["routed_to"] == "fast"
        assert c.post("/admin/mode", json={"mode": "nope"}).status_code == 400


def test_controller_pressure_moves_traffic_to_fast_tier():
    app, fake = make()
    gw = app.state.gateway
    with TestClient(app) as c:
        gw.controller.fast_share = 1.0
        body = c.post("/predict", content=b"img").json()
    assert body["routed_to"] == "fast" and body["reason"] == "sla_pressure"


def test_canary_rollout_rolls_back_a_failing_version():
    app, fake = make(canary=CanaryConfig(stages=(0.5, 1.0), stage_s=0, min_requests=5))
    fake.failing.add("canary")
    gw = app.state.gateway
    with TestClient(app) as c:
        started = c.post("/admin/canary", json={"name": "acc-v2", "url": "http://canary"}).json()
        assert started["state"] == "running" and started["weight"] == 0.5
        for _ in range(40):
            c.post("/predict", content=b"img")
        gw.tick()
        status = c.get("/admin/canary").json()
    assert status["state"] == "rolled_back" and status["reason"] == "canary circuit breaker opened"
    assert gw.canary is None and gw.stable.name == "acc-v1"
    assert fake.calls["canary"] > 0


def test_canary_rollout_promotes_a_healthy_version():
    app, fake = make(canary=CanaryConfig(stages=(0.5, 1.0), stage_s=0, min_requests=5))
    gw = app.state.gateway
    with TestClient(app) as c:
        c.post("/admin/canary", json={"name": "acc-v2", "url": "http://canary"})
        for _ in range(2):
            for _ in range(40):
                c.post("/predict", content=b"img")
            gw.tick()
        status = c.get("/admin/canary").json()
    assert status["state"] == "promoted"
    assert [h["weight"] for h in status["history"]] == [0.5, 1.0, 0.0]
    assert gw.stable.name == "acc-v2" and gw.canary is None


def test_shadow_traffic_records_agreement_without_changing_the_response():
    app, fake = make(shadow=1.0)
    with TestClient(app) as c:
        body = c.post("/predict", content=b"img").json()
        shadow = c.get("/state").json()["shadow"]
    assert body["routed_to"] == "accurate"
    assert fake.calls["fast"] == 1 and shadow == {"agree": 1, "disagree": 0}


@pytest.mark.parametrize("path", ["/healthz", "/metrics", "/state"])
def test_ops_endpoints(path):
    app, _ = make()
    with TestClient(app) as c:
        assert c.get(path).status_code == 200
