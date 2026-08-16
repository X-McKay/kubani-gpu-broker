"""Prometheus metrics.

A per-app CollectorRegistry (rather than the global default) keeps app
instances independent, which matters for tests that build many apps in
one process.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from .state import ENGINE_STATE_VALUES, EngineState


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.inflight_requests = Gauge(
            "kubani_gpu_broker_inflight_requests",
            "In-flight proxied inference requests",
            ["engine"],
            registry=self.registry,
        )
        self.requests_total = Counter(
            "kubani_gpu_broker_requests_total",
            "Proxied inference requests",
            ["engine", "status"],
            registry=self.registry,
        )
        self.upstream_errors_total = Counter(
            "kubani_gpu_broker_upstream_errors_total",
            "Errors reaching the upstream engine",
            ["engine", "type"],
            registry=self.registry,
        )
        self.engine_state = Gauge(
            "kubani_gpu_broker_engine_state",
            "Engine lifecycle state (0=unknown 1=awake 2=waking 3=sleeping 4=error)",
            ["engine"],
            registry=self.registry,
        )
        self.sleep_total = Counter(
            "kubani_gpu_broker_sleep_total",
            "Engine sleep transitions",
            ["engine", "level", "result"],
            registry=self.registry,
        )
        self.wake_total = Counter(
            "kubani_gpu_broker_wake_total",
            "Engine wake transitions",
            ["engine", "result"],
            registry=self.registry,
        )
        self.wake_duration_seconds = Histogram(
            "kubani_gpu_broker_wake_duration_seconds",
            "Wall-clock duration of successful wake transitions",
            ["engine"],
            registry=self.registry,
        )

    def set_engine_state(self, engine: str, state: EngineState) -> None:
        self.engine_state.labels(engine=engine).set(ENGINE_STATE_VALUES[state])
