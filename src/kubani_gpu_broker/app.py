"""Application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .config import BrokerConfig, load_config
from .engines import Engine
from .proxy import build_router
from .telemetry import Metrics


def create_app(
    config: BrokerConfig | None = None,
    client: httpx.AsyncClient | None = None,
) -> FastAPI:
    cfg = config or load_config()

    # Phase A manages a single engine; the registry shape anticipates more.
    name, engine_cfg = next(iter(cfg.engines.items()))
    engine = Engine(name, engine_cfg.base_url)
    metrics = Metrics()

    owns_client = client is None
    http_client = client or httpx.AsyncClient(
        # Unbounded read/write/pool: streaming completions may run for a
        # very long time. Only connecting is bounded (spec section 17).
        timeout=httpx.Timeout(
            connect=cfg.proxy.connect_timeout_seconds, read=None, write=None, pool=None
        ),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            if owns_client:
                await http_client.aclose()

    app = FastAPI(title="kubani-gpu-broker", lifespan=lifespan)
    app.state.engine = engine
    app.state.metrics = metrics

    @app.get("/healthz", response_class=PlainTextResponse)
    async def healthz() -> str:
        return "ok"

    @app.get("/readyz", response_class=PlainTextResponse)
    async def readyz() -> str:
        # Later phases gate readiness on state reconstruction
        # (spec section 22.1); a configured engine suffices for Phase 1.
        return "ok"

    @app.get("/metrics")
    async def metrics_endpoint() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    app.include_router(build_router(engine, http_client, metrics))
    return app
