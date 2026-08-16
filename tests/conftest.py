"""Test fixtures: a fake vLLM upstream and a broker wired to it in-process.

The broker's httpx client is given an ASGITransport pointing at the fake
vLLM app, so requests traverse the real proxy code path with no sockets.

The fake models vLLM sleep-mode semantics, including the property that a
real sleeping vLLM does NOT serve inference (upstream, such requests hang;
here they are recorded as violations and fail loudly) — the tests assert
the broker never lets a request reach a sleeping engine.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from kubani_gpu_broker.admin import create_admin_app
from kubani_gpu_broker.app import Broker, create_public_app
from kubani_gpu_broker.config import (
    BrokerConfig,
    EngineConfig,
    PoliciesConfig,
)

ADMIN_TOKEN = "test-admin-token"


class FakeVllm:
    """Minimal OpenAI-compatible upstream with sleep-mode semantics."""

    def __init__(self) -> None:
        self.app = FastAPI()
        self.requests_seen: list[dict] = []
        self.sleeping = False
        self.sleep_level: int | None = None
        self.sleep_calls = 0
        self.wake_calls = 0
        # Requests that reached inference endpoints while sleeping — must
        # always stay zero; the broker's gate is a correctness requirement.
        self.violations = 0
        # When set, /wake_up blocks until released (slow-wake simulation).
        self.wake_gate: asyncio.Event | None = None
        # When true, /wake_up fails with a 500.
        self.wake_fail = False
        # When true, /wake_up returns OK but the engine stays sleeping
        # (models the GB10 silent-EngineCore-death mode, vllm#50011).
        self.wake_noop = False
        # When set, the streaming endpoint blocks mid-stream until released.
        self.stream_gate: asyncio.Event | None = None
        self._register()

    def _register(self) -> None:
        app = self.app

        @app.get("/is_sleeping")
        async def is_sleeping():
            return {"is_sleeping": self.sleeping}

        @app.post("/sleep")
        async def sleep(level: int = 1):
            self.sleep_calls += 1
            self.sleeping = True
            self.sleep_level = level
            return {"status": "ok"}

        @app.post("/wake_up")
        async def wake_up():
            self.wake_calls += 1
            if self.wake_gate is not None:
                await self.wake_gate.wait()
            if self.wake_fail:
                return JSONResponse({"error": "EngineCore died"}, status_code=500)
            if not self.wake_noop:
                self.sleeping = False
            return {"status": "ok"}

        @app.get("/v1/models")
        async def models():
            if self.sleeping:
                self.violations += 1
                return JSONResponse({"error": "engine is sleeping"}, status_code=500)
            return {"object": "list", "data": [{"id": "fake-model", "object": "model"}]}

        @app.post("/v1/chat/completions")
        async def chat(request: Request):
            if self.sleeping:
                self.violations += 1
                return JSONResponse({"error": "engine is sleeping"}, status_code=500)
            body = await request.json()
            self.requests_seen.append(body)

            if not body.get("stream"):
                return JSONResponse(
                    {
                        "id": "cmpl-1",
                        "object": "chat.completion",
                        "model": body.get("model", "fake-model"),
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "pong"},
                                "finish_reason": "stop",
                            }
                        ],
                    },
                    headers={"x-fake-vllm": "1"},
                )

            async def sse():
                for i in range(3):
                    if self.stream_gate is not None and i == 1:
                        await self.stream_gate.wait()
                    chunk = {"choices": [{"delta": {"content": f"tok{i}"}}]}
                    yield f"data: {json.dumps(chunk)}\n\n".encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        @app.post("/v1/broken")
        async def broken():
            if self.sleeping:
                self.violations += 1
                return JSONResponse({"error": "engine is sleeping"}, status_code=500)
            return JSONResponse({"error": {"type": "bad_request"}}, status_code=400)


def make_config(**engine_overrides) -> BrokerConfig:
    return BrokerConfig(
        engines={"main": EngineConfig(base_url="http://fake-vllm", **engine_overrides)},
        policies=PoliciesConfig(min_awake_seconds=0.0, idle_check_interval_seconds=0.01),
        admin_token=ADMIN_TOKEN,
    )


def make_broker(fake_vllm: FakeVllm, **engine_overrides) -> Broker:
    upstream_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake_vllm.app), base_url="http://fake-vllm"
    )
    return Broker(make_config(**engine_overrides), client=upstream_client)


@pytest.fixture
def fake_vllm() -> FakeVllm:
    return FakeVllm()


@pytest.fixture
def broker(fake_vllm: FakeVllm) -> Broker:
    return make_broker(fake_vllm)


@pytest.fixture
def sleep_broker(fake_vllm: FakeVllm) -> Broker:
    return make_broker(
        fake_vllm,
        sleep_enabled=True,
        idle_timeout_seconds=0.05,
        wake_timeout_seconds=2.0,
        wake_poll_interval_seconds=0.01,
    )


@pytest.fixture
def broker_app(broker: Broker):
    return create_public_app(broker)


@pytest.fixture
async def broker_client(broker_app):
    transport = httpx.ASGITransport(app=broker_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://broker") as client:
        yield client


@pytest.fixture
async def sleep_client(sleep_broker: Broker):
    transport = httpx.ASGITransport(app=create_public_app(sleep_broker))
    async with httpx.AsyncClient(transport=transport, base_url="http://broker") as client:
        yield client


@pytest.fixture
async def admin_client(sleep_broker: Broker):
    transport = httpx.ASGITransport(app=create_admin_app(sleep_broker))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://admin",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    ) as client:
        yield client
