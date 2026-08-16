"""Transparent streaming reverse proxy for /v1/*.

Phase 1 of the adopted spec: byte-faithful passthrough to the engine with
in-flight accounting. No idle sleep, no wake, no lease yet — but the
accounting here is what those phases build on, so its lifetime rules
matter: a request is in flight from just before the upstream send until
the response body is fully delivered (or the client disconnects), never
merely until first byte.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

from .engines import Engine
from .telemetry import Metrics

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


def build_router(engine: Engine, client: httpx.AsyncClient, metrics: Metrics) -> APIRouter:
    router = APIRouter()

    @router.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    )
    async def proxy(request: Request, path: str) -> Response:
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
            return Response(
                content=(
                    '{"error": {"type": "upstream_unreachable", '
                    '"message": "The inference engine could not be reached."}}'
                ),
                status_code=502,
                media_type="application/json",
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


def _finish(engine: Engine, metrics: Metrics) -> None:
    engine.release()
    metrics.inflight_requests.labels(engine=engine.name).dec()
