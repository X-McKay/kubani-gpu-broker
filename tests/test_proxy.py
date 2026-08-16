from __future__ import annotations

import asyncio
import json

import httpx

from kubani_gpu_broker.app import create_app
from kubani_gpu_broker.config import BrokerConfig, EngineConfig


async def test_json_roundtrip(broker_client, fake_vllm):
    resp = await broker_client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": [{"role": "user", "content": "ping"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "pong"
    # Upstream response headers survive the proxy.
    assert resp.headers["x-fake-vllm"] == "1"
    # The upstream saw the request body unmodified.
    assert fake_vllm.requests_seen[0]["messages"][0]["content"] == "ping"


async def test_get_models_passthrough(broker_client):
    resp = await broker_client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json()["data"][0]["id"] == "fake-model"


async def test_streaming_sse(broker_client):
    chunks: list[bytes] = []
    async with broker_client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "fake-model", "stream": True, "messages": []},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        async for chunk in resp.aiter_raw():
            chunks.append(chunk)
    payload = b"".join(chunks)
    assert payload.count(b"data:") == 4  # 3 tokens + [DONE]
    assert b"[DONE]" in payload


async def test_upstream_error_passthrough(broker_client):
    resp = await broker_client.post("/v1/broken", json={})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "bad_request"


async def test_unreachable_upstream_returns_502():
    config = BrokerConfig(engines={"main": EngineConfig(base_url="http://127.0.0.1:1")})
    client = httpx.AsyncClient(timeout=httpx.Timeout(connect=0.2, read=1, write=1, pool=1))
    app = create_app(config=config, client=client)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://broker") as c:
        resp = await c.post("/v1/chat/completions", json={})
    await client.aclose()
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "upstream_unreachable"


async def test_inflight_accounting_spans_stream(broker_app, fake_vllm):
    """A streaming request stays in flight until the stream finishes.

    Spec section 9.2: do not reset the idle clock at first token.
    """
    engine = broker_app.state.engine
    fake_vllm.stream_gate = asyncio.Event()

    transport = httpx.ASGITransport(app=broker_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://broker") as client:
        before = engine.last_activity

        async def consume():
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json={"model": "fake-model", "stream": True, "messages": []},
            ) as resp:
                body = b""
                async for chunk in resp.aiter_raw():
                    body += chunk
                return body

        task = asyncio.create_task(consume())
        # First token has been emitted, stream is gated mid-flight. Polling
        # is deliberate: Engine exposes no notification hook to await.
        while engine.in_flight == 0:  # noqa: ASYNC110
            await asyncio.sleep(0.01)
        assert engine.in_flight == 1
        assert engine.idle_seconds() == 0.0

        fake_vllm.stream_gate.set()
        body = await task

    assert b"[DONE]" in body
    assert engine.in_flight == 0
    assert engine.last_activity > before


async def test_metrics_track_requests(broker_client):
    await broker_client.post("/v1/chat/completions", json={"model": "m", "messages": []})
    metrics = (await broker_client.get("/metrics")).text
    assert 'kubani_gpu_broker_requests_total{engine="main",status="200"} 1.0' in metrics
    assert 'kubani_gpu_broker_inflight_requests{engine="main"} 0.0' in metrics


async def test_request_body_streams_through(broker_client, fake_vllm):
    # Large-ish body exercises the streamed request path end to end.
    big = "x" * 65536
    resp = await broker_client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": [{"role": "user", "content": big}]},
    )
    assert resp.status_code == 200
    assert fake_vllm.requests_seen[-1]["messages"][0]["content"] == big


def test_config_loads_yaml(tmp_path):
    from kubani_gpu_broker.config import load_config

    p = tmp_path / "config.yaml"
    p.write_text(
        json.dumps(
            {
                "engines": {"main": {"base_url": "http://llm-engine:8000"}},
                "proxy": {"connect_timeout_seconds": 5},
            }
        )
    )
    cfg = load_config(p)
    assert cfg.engines["main"].base_url == "http://llm-engine:8000"
    assert cfg.proxy.connect_timeout_seconds == 5
