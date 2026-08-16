"""Prometheus metrics.

A per-app CollectorRegistry (rather than the global default) keeps app
instances independent, which matters for tests that build many apps in
one process.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge


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
