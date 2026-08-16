"""Idle auto-sleep policy (spec section 9). Ticks are driven manually
for determinism; the timing loop itself is trivial."""

from __future__ import annotations

import asyncio

from kubani_gpu_broker.state import EngineState


async def test_idle_engine_sleeps_after_timeout(sleep_broker, sleep_client, fake_vllm):
    # Serve one request, then let the (tiny) idle timeout elapse.
    resp = await sleep_client.post(
        "/v1/chat/completions", json={"model": "fake-model", "messages": []}
    )
    assert resp.status_code == 200

    await asyncio.sleep(0.06)  # idle_timeout_seconds is 0.05 in the fixture
    await sleep_broker.idle_loop.tick()

    assert sleep_broker.engine.state == EngineState.SLEEPING
    assert fake_vllm.sleeping is True
    assert fake_vllm.sleep_level == 1
    assert fake_vllm.sleep_calls == 1


async def test_not_yet_idle_engine_stays_awake(sleep_broker, sleep_client, fake_vllm):
    resp = await sleep_client.post(
        "/v1/chat/completions", json={"model": "fake-model", "messages": []}
    )
    assert resp.status_code == 200

    # Tick immediately: idle_timeout has not elapsed.
    await sleep_broker.idle_loop.tick()
    assert sleep_broker.engine.state == EngineState.AWAKE
    assert fake_vllm.sleep_calls == 0


async def test_min_awake_blocks_early_sleep(sleep_broker, sleep_client, fake_vllm):
    sleep_broker.engine.policies.min_awake_seconds = 60.0
    resp = await sleep_client.post(
        "/v1/chat/completions", json={"model": "fake-model", "messages": []}
    )
    assert resp.status_code == 200

    await asyncio.sleep(0.06)
    await sleep_broker.idle_loop.tick()
    assert sleep_broker.engine.state == EngineState.AWAKE
    assert fake_vllm.sleep_calls == 0


async def test_inflight_stream_blocks_idle_sleep(sleep_broker, sleep_client, fake_vllm):
    """Spec section 15.3: an active stream keeps the engine awake even
    past the idle timeout."""
    # Establish AWAKE state first.
    await sleep_client.post("/v1/chat/completions", json={"model": "m", "messages": []})

    fake_vllm.stream_gate = asyncio.Event()
    engine = sleep_broker.engine

    async def consume():
        async with sleep_client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "fake-model", "stream": True, "messages": []},
        ) as resp:
            async for _ in resp.aiter_raw():
                pass

    task = asyncio.create_task(consume())
    while engine.in_flight == 0:  # noqa: ASYNC110 - no hook to await
        await asyncio.sleep(0.01)

    await asyncio.sleep(0.06)  # idle timeout elapses in wall-clock terms
    await sleep_broker.idle_loop.tick()
    assert engine.state == EngineState.AWAKE
    assert fake_vllm.sleep_calls == 0

    fake_vllm.stream_gate.set()
    await task

    await asyncio.sleep(0.06)
    await sleep_broker.idle_loop.tick()
    assert engine.state == EngineState.SLEEPING


async def test_sleep_wake_cycle_end_to_end(sleep_broker, sleep_client, fake_vllm):
    """Full cycle: request -> idle sleep -> transparent wake -> serve."""
    await sleep_client.post("/v1/chat/completions", json={"model": "m", "messages": []})
    await asyncio.sleep(0.06)
    await sleep_broker.idle_loop.tick()
    assert sleep_broker.engine.state == EngineState.SLEEPING

    resp = await sleep_client.post(
        "/v1/chat/completions", json={"model": "fake-model", "messages": []}
    )
    assert resp.status_code == 200
    assert sleep_broker.engine.state == EngineState.AWAKE
    assert fake_vllm.wake_calls == 1
    assert fake_vllm.violations == 0
