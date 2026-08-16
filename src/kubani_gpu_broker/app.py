"""Broker assembly and the public application."""

from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from .config import BrokerConfig, load_config
from .engines import Engine
from .idle import IdleLoop
from .proxy import build_router
from .state import GpuOwnership
from .telemetry import Metrics


class Broker:
    """Owns the shared state both listeners operate on."""

    def __init__(self, cfg: BrokerConfig, client: httpx.AsyncClient | None = None) -> None:
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
        if self.cfg.policies.auto_sleep_enabled:
            self.idle_loop.start()

    async def shutdown(self) -> None:
        await self.idle_loop.stop()
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
