"""Internal admin listener (spec sections 16.3 and 24.2).

Served on its own port, never routed through Ingress. /metrics and
/healthz are unauthenticated (Prometheus and probes reach them via
NetworkPolicy); everything under /internal/ requires the bearer token
from a SOPS-managed Secret and fails closed when none is configured
(kubani spec amendment: admin auth is mandatory, not optional).
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from .lease import (
    DrainTimeoutError,
    JobsStillActiveError,
    LeaseConflictError,
    LeaseNotFoundError,
    LeaseUnavailableError,
)
from .state import EngineBusyError, EngineUnavailableError, WakeFailedError

if TYPE_CHECKING:
    from .app import Broker


class LeaseRequest(BaseModel):
    owner: str = Field(min_length=1)
    workload_id: str = Field(min_length=1)
    reclaim: str = Field(default="sleep", pattern="^(sleep|restart)$")
    drain_timeout_seconds: float | None = None


def create_admin_app(broker: Broker) -> FastAPI:
    app = FastAPI(title="kubani-gpu-broker admin")
    app.state.broker = broker

    def require_token(request: Request) -> None:
        token = broker.cfg.admin_token
        if not token:
            raise HTTPException(status_code=401, detail="admin token not configured")
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer ") or not secrets.compare_digest(
            auth.removeprefix("Bearer "), token
        ):
            raise HTTPException(status_code=401, detail="invalid admin token")

    @app.get("/healthz", response_class=PlainTextResponse)
    async def healthz() -> str:
        return "ok"

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(broker.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    def _engine_or_404(name: str):
        engine = broker.engines.get(name)
        if engine is None:
            raise HTTPException(status_code=404, detail=f"unknown engine: {name}")
        return engine

    @app.get("/internal/v1/state", dependencies=[Depends(require_token)])
    async def state() -> dict:
        return {
            "gpu_owner": broker.ownership.state.value,
            "engines": {name: e.snapshot() for name, e in broker.engines.items()},
            "lease": broker.lease_manager.snapshot(),
        }

    @app.get("/internal/v1/engines", dependencies=[Depends(require_token)])
    async def engines() -> dict:
        return {name: e.snapshot() for name, e in broker.engines.items()}

    @app.post(
        "/internal/v1/engines/{name}/sleep",
        dependencies=[Depends(require_token)],
        status_code=202,
    )
    async def sleep_engine(name: str, level: int | None = None) -> dict:
        engine = _engine_or_404(name)
        try:
            await engine.sleep(level=level, manual=True)
        except EngineBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (EngineUnavailableError, WakeFailedError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return engine.snapshot()

    def _lease_payload(lease) -> dict:
        return {
            "lease_id": lease.lease_id,
            "state": "granted",
            "holder": lease.holder,
            "reclaim": lease.reclaim,
            "renewed_at": lease.renewed_at.isoformat(),
        }

    @app.post(
        "/internal/v1/gpu/leases",
        dependencies=[Depends(require_token)],
        status_code=201,
    )
    async def acquire_lease(body: LeaseRequest) -> dict:
        try:
            lease = await broker.lease_manager.acquire(
                owner=body.owner,
                workload_id=body.workload_id,
                reclaim=body.reclaim,
                drain_timeout_seconds=body.drain_timeout_seconds,
            )
        except LeaseConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except DrainTimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except LeaseUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return _lease_payload(lease)

    @app.post(
        "/internal/v1/gpu/leases/{lease_id}/renew",
        dependencies=[Depends(require_token)],
    )
    async def renew_lease(lease_id: str) -> dict:
        try:
            lease = await broker.lease_manager.renew(lease_id)
        except LeaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except LeaseUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return _lease_payload(lease)

    @app.delete(
        "/internal/v1/gpu/leases/{lease_id}",
        dependencies=[Depends(require_token)],
        status_code=204,
    )
    async def release_lease(lease_id: str) -> None:
        try:
            await broker.lease_manager.release(lease_id)
        except LeaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except JobsStillActiveError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except LeaseUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post(
        "/internal/v1/engines/{name}/wake",
        dependencies=[Depends(require_token)],
        status_code=202,
    )
    async def wake_engine(name: str) -> dict:
        engine = _engine_or_404(name)
        # Admin wake is the recovery path: allow retrying out of ERROR.
        await engine.reset_error()
        try:
            await engine.ensure_awake()
        except (WakeFailedError, EngineUnavailableError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return engine.snapshot()

    return app
