"""Broker assembly and the public application."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from .config import BrokerConfig, load_config
from .engines import Engine
from .idle import IdleLoop
from .kube import InClusterKubeClient, KubeClient
from .lease import LeaseManager
from .proxy import build_router
from .state import GpuOwnership
from .telemetry import Metrics


class Broker:
    """Owns the shared state both listeners operate on."""

    def __init__(
        self,
        cfg: BrokerConfig,
        client: httpx.AsyncClient | None = None,
        kube: KubeClient | None = None,
    ) -> None:
        self.cfg = cfg
        self.metrics = Metrics()
        self.ownership = GpuOwnership()
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            # Unbounded read/write/pool: streaming completions may run for a
            # very long time. Only connecting is bounded (spec section 17).
            timeout=httpx.Timeout(
                connect=cfg.proxy.connect_timeout_seconds, read=None, write=None, pool=None
            ),
        )
        self.engines = {
            name: Engine(name, engine_cfg, cfg.policies, self.client, self.metrics, self.ownership)
            for name, engine_cfg in cfg.engines.items()
        }
        self.idle_loop = IdleLoop(self.engines, cfg.policies.idle_check_interval_seconds)
        if kube is None and cfg.gpu.leases_enabled:
            kube = InClusterKubeClient(cfg.gpu.lease_name, cfg.gpu.lease_namespace)
        self.lease_manager = LeaseManager(
            self.ownership,
            self.engines,
            kube,
            duration_seconds=cfg.gpu.lease_duration_seconds,
            drain_timeout_seconds=cfg.policies.drain_timeout_seconds,
        )
        self._lease_watchdog: asyncio.Task | None = None

    @property
    def engine(self) -> Engine:
        # Phase A routes everything to the single configured engine.
        return next(iter(self.engines.values()))

    async def startup(self) -> None:
        for engine in self.engines.values():
            try:
                await engine.refresh_state()
            except Exception:
                # Leave the engine UNKNOWN; ensure_awake() resolves it on
                # first use. Readiness does not depend on the engine being
                # reachable — the broker can hold the fort and 502.
                pass
        await self.lease_manager.reconstruct()
        if self.cfg.policies.auto_sleep_enabled:
            self.idle_loop.start()
        if self.lease_manager.enabled:
            self._lease_watchdog = asyncio.get_running_loop().create_task(
                self._watch_lease_expiry()
            )

    async def _watch_lease_expiry(self) -> None:
        interval = max(self.cfg.gpu.lease_duration_seconds / 3, 1)
        while True:
            await asyncio.sleep(interval)
            await self.lease_manager.tick()

    async def shutdown(self) -> None:
        await self.idle_loop.stop()
        if self._lease_watchdog is not None:
            self._lease_watchdog.cancel()
            try:
                await self._lease_watchdog
            except asyncio.CancelledError:
                pass
            self._lease_watchdog = None
        if self._owns_client:
            await self.client.aclose()


def create_public_app(broker: Broker) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await broker.startup()
        try:
            yield
        finally:
            await broker.shutdown()

    app = FastAPI(title="kubani-gpu-broker", lifespan=lifespan)
    app.state.broker = broker

    @app.get("/healthz", response_class=PlainTextResponse)
    async def healthz() -> str:
        return "ok"

    @app.get("/readyz", response_class=PlainTextResponse)
    async def readyz() -> str:
        # Later phases gate readiness on lease-state reconstruction
        # (spec section 22.1); a configured engine suffices for now.
        return "ok"

    app.include_router(build_router(broker))
    return app


def create_app(
    config: BrokerConfig | None = None,
    client: httpx.AsyncClient | None = None,
) -> FastAPI:
    """Factory for the public listener (uvicorn --factory entrypoint)."""
    return create_public_app(Broker(config or load_config(), client))
