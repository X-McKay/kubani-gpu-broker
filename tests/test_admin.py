"""Admin listener: auth (fail closed), state, and manual sleep/wake."""

from __future__ import annotations

import asyncio

import httpx

from conftest import make_broker
from kubani_gpu_broker.admin import create_admin_app
from kubani_gpu_broker.state import EngineState


async def test_admin_requires_token(sleep_broker):
    transport = httpx.ASGITransport(app=create_admin_app(sleep_broker))
    async with httpx.AsyncClient(transport=transport, base_url="http://admin") as anon:
        assert (await anon.get("/internal/v1/state")).status_code == 401
        assert (await anon.post("/internal/v1/engines/main/sleep")).status_code == 401
        # Wrong token is also rejected.
        bad = await anon.get("/internal/v1/state", headers={"Authorization": "Bearer wrong"})
        assert bad.status_code == 401


async def test_admin_fails_closed_without_configured_token(fake_vllm):
    broker = make_broker(fake_vllm, sleep_enabled=True)
    broker.cfg.admin_token = None
    transport = httpx.ASGITransport(app=create_admin_app(broker))
    async with httpx.AsyncClient(transport=transport, base_url="http://admin") as anon:
        resp = await anon.get("/internal/v1/state", headers={"Authorization": "Bearer anything"})
    assert resp.status_code == 401
    assert "not configured" in resp.json()["detail"]
    await broker.client.aclose()


async def test_metrics_and_healthz_unauthenticated(sleep_broker):
    transport = httpx.ASGITransport(app=create_admin_app(sleep_broker))
    async with httpx.AsyncClient(transport=transport, base_url="http://admin") as anon:
        assert (await anon.get("/healthz")).status_code == 200
        metrics = await anon.get("/metrics")
    assert metrics.status_code == 200
    assert "kubani_gpu_broker_engine_state" in metrics.text


async def test_state_endpoint(admin_client):
    resp = await admin_client.get("/internal/v1/state")
    assert resp.status_code == 200
    body = resp.json()
    assert body["gpu_owner"] == "AVAILABLE"
    assert body["engines"]["main"]["sleep_enabled"] is True


async def test_manual_sleep_and_wake(admin_client, sleep_broker, fake_vllm):
    resp = await admin_client.post("/internal/v1/engines/main/sleep", params={"level": 1})
    assert resp.status_code == 202
    assert resp.json()["state"] == "sleeping"
    assert fake_vllm.sleep_level == 1

    resp = await admin_client.post("/internal/v1/engines/main/wake")
    assert resp.status_code == 202
    assert resp.json()["state"] == "awake"
    assert sleep_broker.engine.state == EngineState.AWAKE


async def test_manual_sleep_refuses_while_busy(admin_client, sleep_client, fake_vllm):
    fake_vllm.stream_gate = asyncio.Event()

    async def consume():
        async with sleep_client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "fake-model", "stream": True, "messages": []},
        ) as resp:
            async for _ in resp.aiter_raw():
                pass

    task = asyncio.create_task(consume())
    while fake_vllm.stream_gate is not None and not fake_vllm.requests_seen:  # noqa: ASYNC110
        await asyncio.sleep(0.01)

    resp = await admin_client.post("/internal/v1/engines/main/sleep")
    assert resp.status_code == 409

    fake_vllm.stream_gate.set()
    await task


async def test_unknown_engine_404(admin_client):
    resp = await admin_client.post("/internal/v1/engines/nope/sleep")
    assert resp.status_code == 404
