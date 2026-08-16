"""Broker configuration.

Loaded from a YAML file (CONFIG_PATH env var or an explicit path). The
schema mirrors the adopted spec in the kubani repo:
docs/plans/active/2026-08-16-vllm-gpu-sleep-wake-broker.md

The admin bearer token is intentionally NOT part of the YAML file (which
lives in a ConfigMap): it comes from the GPU_BROKER_ADMIN_TOKEN
environment variable, sourced from a SOPS-managed Secret. Admin
endpoints fail closed (401) when no token is configured.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel


class EngineConfig(BaseModel):
    base_url: str
    sleep_enabled: bool = False
    idle_timeout_seconds: float = 600.0
    idle_sleep_level: int = 1
    wake_timeout_seconds: float = 120.0
    wake_poll_interval_seconds: float = 1.0


class PoliciesConfig(BaseModel):
    auto_sleep_enabled: bool = False
    min_awake_seconds: float = 120.0
    idle_check_interval_seconds: float = 10.0
    drain_timeout_seconds: float = 300.0


class GpuConfig(BaseModel):
    leases_enabled: bool = False
    lease_name: str = "sparky-gpu"
    lease_namespace: str = "vllm"
    lease_duration_seconds: int = 180


class ProxyConfig(BaseModel):
    # Connect timeout only. Read/write/pool are unbounded: the broker must
    # support arbitrarily long streaming completions (spec section 17).
    connect_timeout_seconds: float = 10.0


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    public_port: int = 8080
    admin_port: int = 8081


class BrokerConfig(BaseModel):
    engines: dict[str, EngineConfig]
    policies: PoliciesConfig = PoliciesConfig()
    gpu: GpuConfig = GpuConfig()
    proxy: ProxyConfig = ProxyConfig()
    server: ServerConfig = ServerConfig()
    admin_token: str | None = None


def load_config(path: str | Path | None = None) -> BrokerConfig:
    resolved = Path(path or os.environ["CONFIG_PATH"])
    with resolved.open() as f:
        raw = yaml.safe_load(f)
    cfg = BrokerConfig.model_validate(raw)
    token = os.environ.get("GPU_BROKER_ADMIN_TOKEN")
    if token:
        cfg.admin_token = token
    return cfg
