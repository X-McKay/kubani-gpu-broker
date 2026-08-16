"""Broker configuration.

Loaded from a YAML file (CONFIG_PATH env var or an explicit path). The
schema mirrors the adopted spec in the kubani repo:
docs/plans/active/2026-08-16-vllm-gpu-sleep-wake-broker.md
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel


class EngineConfig(BaseModel):
    base_url: str


class ProxyConfig(BaseModel):
    # Connect timeout only. Read/write/pool are unbounded: the broker must
    # support arbitrarily long streaming completions (spec section 17).
    connect_timeout_seconds: float = 10.0


class BrokerConfig(BaseModel):
    engines: dict[str, EngineConfig]
    proxy: ProxyConfig = ProxyConfig()


def load_config(path: str | Path | None = None) -> BrokerConfig:
    resolved = Path(path or os.environ["CONFIG_PATH"])
    with resolved.open() as f:
        raw = yaml.safe_load(f)
    return BrokerConfig.model_validate(raw)
