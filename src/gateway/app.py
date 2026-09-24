"""SLA-aware inference gateway.

POST /predict  image bytes in, prediction out, plus which tier served it and why.

Every second a control loop updates the fast-tier share (SlaController), the
per-tier concurrency limits (AdaptiveLimit) and any canary rollout. Each
request is then routed by `routing.decide` using only in-memory state, so the
gateway adds no extra network calls on the request path.
"""

from __future__ import annotations

import asyncio
import random
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from . import config as config_module
from .breaker import CLOSED, HALF_OPEN, OPEN, CircuitBreaker
from .canary import PROMOTED, RUNNING, CanaryRollout
from .controller import AdaptiveLimit, SlaController
from .routing import ACCURATE, FAST, MODES, SLA, TierView, decide
from .stats import RollingWindow

BUCKETS = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 30)
REQUESTS = Counter("gateway_requests_total", "Requests by tier, reason and outcome", ("tier", "reason", "outcome"))
LATENCY = Histogram("gateway_request_seconds", "End-to-end latency through the gateway", ("tier",), buckets=BUCKETS)
FAST_SHARE = Gauge("gateway_fast_share", "Share of traffic the controller sends to the fast tier")
LIMIT = Gauge("gateway_concurrency_limit", "Adaptive concurrency limit", ("tier",))
IN_FLIGHT = Gauge("gateway_in_flight", "Outstanding requests", ("tier",))
BREAKER = Gauge("gateway_breaker_state", "0 closed, 1 half-open, 2 open", ("target",))
CANARY_WEIGHT = Gauge("gateway_canary_weight", "Share of accurate-tier traffic on the canary")
TIER_P95 = Gauge("gateway_tier_p95_seconds", "Rolling p95 latency the controller sees", ("tier",))
SHADOW = Counter("gateway_shadow_total", "Shadow comparisons of fast vs accurate top-1", ("agree",))


@dataclass
class Backend:
    name: str
    url: str
    tier: str
    breaker: CircuitBreaker
    window: RollingWindow = field(default_factory=lambda: RollingWindow(30.0))
    in_flight: int = 0


class Gateway:
    def __init__(self, cfg: config_module.GatewayConfig) -> None:
        self.cfg = cfg
        self.mode = cfg.mode
        self.client: httpx.AsyncClient | None = None
        self.stable = self._backend(cfg.accurate.target, ACCURATE)
        self.fast = self._backend(cfg.fast.target, FAST)
        self.canary: Backend | None = None
        self.rollout = CanaryRollout(cfg.canary)
        self.shadow = {"agree": 0, "disagree": 0}
        self._background: set[asyncio.Task] = set()  # keep references so tasks are not garbage-collected
        self.reset()

    def _backend(self, target: config_module.Target, tier: str) -> Backend:
        return Backend(target.name, target.url, tier,
                       CircuitBreaker(self.cfg.breaker_failures, self.cfg.breaker_open_s))

    def reset(self) -> None:
        """Clear learned state (used between benchmark runs)."""
        c = self.cfg
        self.controller = SlaController(c.controller)
        self.limits = {ACCURATE: AdaptiveLimit(c.accurate.limit, c.controller),
                       FAST: AdaptiveLimit(c.fast.limit, c.controller)}
        self.tier_windows = {ACCURATE: RollingWindow(c.controller.window_s),
                             FAST: RollingWindow(c.controller.window_s)}
        for b in self.backends():
            b.window.clear()
            b.breaker = CircuitBreaker(c.breaker_failures, c.breaker_open_s)
        self.shadow = {"agree": 0, "disagree": 0}

    def backends(self) -> list[Backend]:
        return [b for b in (self.stable, self.canary, self.fast) if b is not None]

    # ---- request path -------------------------------------------------------

    def _tier_view(self, tier: str, now: float) -> TierView:
        members = [self.stable, self.canary] if tier == ACCURATE else [self.fast]
        members = [b for b in members if b is not None]
        limit = self.limits[tier].limit if self.mode == SLA else 10**9
        return TierView(in_flight=sum(b.in_flight for b in members), max_in_flight=limit,
                        breaker_allows=any(b.breaker.peek(now) for b in members))

    def _pick_accurate_backend(self) -> Backend:
        if self.canary is not None and random.random() < self.rollout.weight:
            return self.canary
        return self.stable

    async def _call(self, backend: Backend, body: bytes, timeout: float) -> tuple[int, dict]:
        assert self.client is not None
        backend.in_flight += 1
        start = time.monotonic()
        try:
            r = await self.client.post(f"{backend.url}/predict", content=body, timeout=timeout)
            status = r.status_code
            data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        except httpx.TimeoutException:
            status, data = 504, {"detail": "backend timeout"}
        except httpx.TransportError as exc:
            status, data = 502, {"detail": f"backend unreachable: {type(exc).__name__}"}
        finally:
            backend.in_flight -= 1
        now = time.monotonic()
        latency_ms = (now - start) * 1000
        healthy = status < 500  # a 4xx is the client's fault, not the backend's
        backend.breaker.record(now, healthy)
        backend.window.add(now, latency_ms, status == 200)
        self.tier_windows[backend.tier].add(now, latency_ms, status == 200)
        return status, data

    async def predict(self, body: bytes) -> tuple[int, dict]:
        start = time.monotonic()
        acc_view, fast_view = self._tier_view(ACCURATE, start), self._tier_view(FAST, start)
        decision = decide(self.mode, self.controller.fast_share, random.random(), acc_view, fast_view)
        if self.mode == SLA:
            if acc_view.saturated:
                self.limits[ACCURATE].note_saturated()
            if fast_view.saturated:
                self.limits[FAST].note_saturated()
        if decision.tier is None:
            return self._finish(start, None, decision.reason, 503, {"detail": "overloaded, try again"})

        timeout = self.cfg.request_timeout_s if self.mode == SLA else self.cfg.baseline_timeout_s
        backend = self._pick_accurate_backend() if decision.tier == ACCURATE else self.fast
        reason = decision.reason
        if self.mode == SLA and not backend.breaker.allow(start):
            status, data = 503, {"detail": "circuit open"}
        else:
            status, data = await self._call(backend, body, timeout)

        # One fallback attempt on the fast tier if the accurate tier failed.
        if self.mode == SLA and status >= 500 and backend.tier == ACCURATE:
            remaining = timeout - (time.monotonic() - start)
            if remaining > 0.05 and self._tier_view(FAST, time.monotonic()).available \
                    and self.fast.breaker.allow(time.monotonic()):
                backend, reason = self.fast, "fallback_after_error"
                status, data = await self._call(backend, body, remaining)

        if status == 200 and backend.tier == ACCURATE and self.cfg.shadow_fraction > 0 \
                and random.random() < self.cfg.shadow_fraction and self._tier_view(FAST, start).available:
            task = asyncio.create_task(self._shadow(body, data))
            self._background.add(task)
            task.add_done_callback(self._background.discard)
        return self._finish(start, backend, reason, status, data)

    def _finish(self, start: float, backend: Backend | None, reason: str, status: int, data: dict):
        tier = backend.tier if backend else "none"
        outcome = "ok" if status == 200 else ("shed" if backend is None else "error")
        REQUESTS.labels(tier, reason, outcome).inc()
        elapsed = time.monotonic() - start
        LATENCY.labels(tier).observe(elapsed)
        data = {**data, "routed_to": tier, "target": backend.name if backend else None,
                "reason": reason, "gateway_ms": round(elapsed * 1000, 2)}
        return status, data

    async def _shadow(self, body: bytes, primary: dict) -> None:
        """Send a copy to the fast tier and record whether its top-1 agrees. Never affects the client."""
        assert self.client is not None
        self.fast.in_flight += 1
        try:
            r = await self.client.post(f"{self.fast.url}/predict", content=body, timeout=self.cfg.request_timeout_s)
            if r.status_code == 200:
                agree = r.json()["top5"][0]["index"] == primary["top5"][0]["index"]
                self.shadow["agree" if agree else "disagree"] += 1
                SHADOW.labels(str(agree).lower()).inc()
        except httpx.HTTPError:
            pass
        finally:
            self.fast.in_flight -= 1

    # ---- control loop -------------------------------------------------------

    def tick(self) -> None:
        now = time.monotonic()
        acc = self.tier_windows[ACCURATE].snapshot(now)
        fast = self.tier_windows[FAST].snapshot(now)
        if self.mode == SLA:
            self.controller.tick(now, acc)
            for tier, snap in ((ACCURATE, acc), (FAST, fast)):
                tier_cfg = self.cfg.accurate if tier == ACCURATE else self.cfg.fast
                if tier_cfg.adaptive:
                    self.limits[tier].tick(now, snap)
        if self.canary is not None and self.rollout.state == RUNNING:
            if self.canary.breaker.state == OPEN:
                state = self.rollout.abort(now, "canary circuit breaker opened")
            else:
                state = self.rollout.tick(now, self.stable.window.snapshot(now), self.canary.window.snapshot(now))
            if state == PROMOTED:
                self.stable, self.canary = self.canary, None
            elif state != RUNNING:
                self.canary = None
        self._export_gauges(now, acc, fast)

    def _export_gauges(self, now: float, acc, fast) -> None:
        FAST_SHARE.set(self.controller.fast_share if self.mode == SLA else float(self.mode == "always_fast"))
        for tier, snap in ((ACCURATE, acc), (FAST, fast)):
            LIMIT.labels(tier).set(self.limits[tier].limit)
            IN_FLIGHT.labels(tier).set(self._tier_view(tier, now).in_flight)
            if snap.p95_ms is not None:
                TIER_P95.labels(tier).set(snap.p95_ms / 1000)
        for b in self.backends():
            BREAKER.labels(b.name).set({CLOSED: 0, HALF_OPEN: 1}.get(b.breaker.state, 2))
        CANARY_WEIGHT.set(self.rollout.weight)

    def state(self) -> dict:
        now = time.monotonic()
        snap = lambda w: asdict(w.snapshot(now))  # noqa: E731
        return {
            "mode": self.mode,
            "fast_share": self.controller.fast_share,
            "controller_reason": self.controller.last_reason,
            "limits": {t: l.limit for t, l in self.limits.items()},
            "tiers": {t: {"window": snap(w), "in_flight": self._tier_view(t, now).in_flight}
                      for t, w in self.tier_windows.items()},
            "backends": {b.name: {"tier": b.tier, "url": b.url, "breaker": b.breaker.state,
                                  "in_flight": b.in_flight, "window_30s": snap(b.window)}
                         for b in self.backends()},
            "canary": self.canary_status(),
            "shadow": self.shadow,
        }

    def canary_status(self) -> dict:
        return {"state": self.rollout.state, "stage": self.rollout.stage, "weight": self.rollout.weight,
                "reason": self.rollout.reason, "stable": self.stable.name,
                "canary": self.canary.name if self.canary else None, "history": self.rollout.history}

    def start_canary(self, name: str, url: str) -> None:
        if self.rollout.state == RUNNING:
            raise HTTPException(409, "a rollout is already running")
        self.canary = Backend(name, url, ACCURATE, CircuitBreaker(self.cfg.breaker_failures, self.cfg.breaker_open_s))
        self.stable.window.clear()
        self.rollout = CanaryRollout(self.cfg.canary)
        self.rollout.start(time.monotonic())


def create_app(cfg: config_module.GatewayConfig | None = None,
               transport: httpx.AsyncBaseTransport | None = None, tick_s: float = 1.0) -> FastAPI:
    """`transport` lets tests plug in fake backends; `tick_s` <= 0 disables the control loop."""
    gw = Gateway(cfg or config_module.load())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        limits = httpx.Limits(max_connections=512, max_keepalive_connections=128)
        gw.client = httpx.AsyncClient(limits=limits, transport=transport)

        async def loop() -> None:
            while True:
                await asyncio.sleep(tick_s)
                gw.tick()

        task = asyncio.create_task(loop()) if tick_s > 0 else None
        yield
        if task:
            task.cancel()
        await gw.client.aclose()

    app = FastAPI(title="SLA-aware inference gateway", lifespan=lifespan)
    app.state.gateway = gw

    @app.post("/predict")
    async def predict(request: Request):
        body = await request.body()
        if not body:
            raise HTTPException(400, "send the image bytes as the request body")
        status, data = await gw.predict(body)
        return JSONResponse(data, status_code=status, headers={"X-Routed-To": data["routed_to"]})

    @app.get("/state")
    def state() -> dict:
        return gw.state()

    @app.post("/admin/mode")
    def set_mode(payload: dict) -> dict:
        if payload.get("mode") not in MODES:
            raise HTTPException(400, f"mode must be one of {MODES}")
        gw.mode = payload["mode"]
        gw.reset()
        return {"mode": gw.mode}

    @app.post("/admin/reset")
    def reset() -> dict:
        gw.reset()
        return {"status": "reset"}

    @app.post("/admin/canary")
    def start_canary(payload: dict) -> dict:
        gw.start_canary(payload["name"], payload["url"])
        return gw.canary_status()

    @app.get("/admin/canary")
    def canary() -> dict:
        return gw.canary_status()

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
