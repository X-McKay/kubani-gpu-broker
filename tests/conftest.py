"""Test fixtures: a fake vLLM upstream and a broker wired to it in-process.

The broker's httpx client is given an ASGITransport pointing at the fake
vLLM app, so requests traverse the real proxy code path with no sockets.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from kubani_gpu_broker.app import create_app
from kubani_gpu_broker.config import BrokerConfig, EngineConfig


class FakeVllm:
    """Minimal OpenAI-compatible upstream with controllable streaming."""

    def __init__(self) -> None:
        self.app = FastAPI()
        self.requests_seen: list[dict] = []
        # When set, the streaming endpoint blocks mid-stream until released.
        self.stream_gate: asyncio.Event | None = None
        self._register()

    def _register(self) -> None:
        app = self.app

        @app.get("/v1/models")
        async def models():
            return {"object": "list", "data": [{"id": "fake-model", "object": "model"}]}

        @app.post("/v1/chat/completions")
        async def chat(request: Request):
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
            return JSONResponse({"error": {"type": "bad_request"}}, status_code=400)


@pytest.fixture
def fake_vllm() -> FakeVllm:
    return FakeVllm()


@pytest.fixture
def broker_app(fake_vllm: FakeVllm):
    config = BrokerConfig(engines={"main": EngineConfig(base_url="http://fake-vllm")})
    upstream_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake_vllm.app), base_url="http://fake-vllm"
    )
    return create_app(config=config, client=upstream_client)


@pytest.fixture
async def broker_client(broker_app):
    transport = httpx.ASGITransport(app=broker_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://broker") as client:
        yield client
