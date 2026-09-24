"""Gateway configuration: a YAML model registry plus tuning knobs.

Environment variables override the backend URLs, so the same file works in
Docker Compose and in Kubernetes:  ACCURATE_URL, FAST_URL.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .canary import CanaryConfig
from .controller import ControllerConfig, LimitConfig


@dataclass
class Target:
    name: str
    url: str


@dataclass
class TierConfig:
    target: Target
    limit: LimitConfig = field(default_factory=LimitConfig)
    adaptive: bool = True


@dataclass
class GatewayConfig:
    accurate: TierConfig
    fast: TierConfig
    mode: str = "sla"
    request_timeout_s: float = 5.0
    baseline_timeout_s: float = 30.0
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    breaker_failures: int = 5
    breaker_open_s: float = 5.0
    canary: CanaryConfig = field(default_factory=CanaryConfig)
    shadow_fraction: float = 0.0


def _tier(raw: dict, url_env: str) -> TierConfig:
    target = Target(**raw["target"])
    target.url = os.environ.get(url_env, target.url)
    return TierConfig(target=target, limit=LimitConfig(**raw.get("limit", {})),
                      adaptive=raw.get("adaptive", True))


def load(path: str | Path | None = None) -> GatewayConfig:
    path = Path(path or os.environ.get("GATEWAY_CONFIG", "deploy/registry.yaml"))
    raw = yaml.safe_load(path.read_text())
    canary = dict(raw.get("canary", {}))
    if "stages" in canary:
        canary["stages"] = tuple(canary["stages"])
    return GatewayConfig(
        accurate=_tier(raw["tiers"]["accurate"], "ACCURATE_URL"),
        fast=_tier(raw["tiers"]["fast"], "FAST_URL"),
        mode=os.environ.get("GATEWAY_MODE", raw.get("mode", "sla")),
        request_timeout_s=raw.get("request_timeout_s", 5.0),
        baseline_timeout_s=raw.get("baseline_timeout_s", 30.0),
        controller=ControllerConfig(**raw.get("controller", {})),
        breaker_failures=raw.get("breaker", {}).get("failure_threshold", 5),
        breaker_open_s=raw.get("breaker", {}).get("open_s", 5.0),
        canary=CanaryConfig(**canary),
        shadow_fraction=float(os.environ.get("SHADOW_FRACTION", raw.get("shadow_fraction", 0.0))),
    )
