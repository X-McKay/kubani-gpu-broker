"""Transparent streaming reverse proxy for /v1/*.

Byte-faithful passthrough with in-flight accounting, fronted by the
sleep/wake gate: a request that finds the engine sleeping triggers a
single-flight wake and is then forwarded — vLLM itself would let the
request hang forever (vllm#45326), so this gate is a correctness
requirement, not a convenience.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

from .state import EngineUnavailableError, GpuUnavailableError, WakeFailedError

if TYPE_CHECKING:
    from .app import Broker

# Hop-by-hop headers must not be forwarded in either direction (RFC 9110).
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",  # recomputed by the server for the possibly re-framed body
}


def _forwardable(headers: httpx.Headers | list[tuple[str, str]]) -> dict[str, str]:
    items = headers.items() if isinstance(headers, httpx.Headers) else headers
    return {k: v for k, v in items if k.lower() not in _HOP_BY_HOP}


def _error_response(status: int, error_type: str, message: str, **headers: str) -> Response:
    return Response(
        content=json.dumps({"error": {"type": error_type, "message": message}}),
        status_code=status,
        media_type="application/json",
        headers=headers or None,
    )


def build_router(broker: Broker) -> APIRouter:
    router = APIRouter()
    engine = broker.engine
    client = broker.client
    metrics = broker.metrics

    @router.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    )
    async def proxy(request: Request, path: str) -> Response:
        try:
            await engine.ensure_awake()
        except GpuUnavailableError:
            metrics.requests_total.labels(engine=engine.name, status="gpu_unavailable").inc()
            return _error_response(
                503,
                "gpu_temporarily_unavailable",
                "The inference GPU is temporarily allocated to an exclusive workload.",
                **{"Retry-After": "30"},
            )
        except (WakeFailedError, EngineUnavailableError):
            metrics.requests_total.labels(engine=engine.name, status="engine_unavailable").inc()
            return _error_response(
                503,
                "engine_unavailable",
                "The inference engine failed to wake and needs recovery.",
                **{"Retry-After": "60"},
            )

        url = f"{engine.base_url}/v1/{path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"

        upstream_request = client.build_request(
            request.method,
            url,
            headers=_forwardable(list(request.headers.items())),
            content=request.stream(),
        )

        engine.acquire()
        metrics.inflight_requests.labels(engine=engine.name).inc()
        try:
            upstream = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            _finish(engine, metrics)
            metrics.upstream_errors_total.labels(engine=engine.name, type=type(exc).__name__).inc()
            metrics.requests_total.labels(engine=engine.name, status="upstream_error").inc()
            return _error_response(
                502, "upstream_unreachable", "The inference engine could not be reached."
            )

        metrics.requests_total.labels(engine=engine.name, status=str(upstream.status_code)).inc()

        async def body() -> AsyncIterator[bytes]:
            # The finally block is the single release point for the happy
            # path, client disconnects (generator cancellation), and
            # mid-stream upstream failures alike.
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()
                _finish(engine, metrics)

        return StreamingResponse(
            body(),
            status_code=upstream.status_code,
            headers=_forwardable(upstream.headers),
        )

    return router


def _finish(engine, metrics) -> None:
    engine.release()
    metrics.inflight_requests.labels(engine=engine.name).dec()
