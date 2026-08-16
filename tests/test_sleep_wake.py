"""Sleep/wake lifecycle: transparent wake, single-flight, failure, recovery.

The invariant asserted throughout: fake_vllm.violations == 0 — no request
ever reaches a sleeping engine (a real vLLM would hang it, vllm#45326).
"""

from __future__ import annotations

import asyncio

from kubani_gpu_broker.state import EngineState


async def _sleep_engine(broker, fake_vllm):
    await broker.engine.sleep(manual=True)
    assert fake_vllm.sleeping is True
    assert broker.engine.state == EngineState.SLEEPING


async def test_transparent_wake_on_request(sleep_broker, sleep_client, fake_vllm):
    await _sleep_engine(sleep_broker, fake_vllm)

    resp = await sleep_client.post(
        "/v1/chat/completions", json={"model": "fake-model", "messages": []}
    )

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "pong"
    assert fake_vllm.wake_calls == 1
    assert fake_vllm.violations == 0
    assert sleep_broker.engine.state == EngineState.AWAKE


async def test_concurrent_requests_single_flight_wake(sleep_broker, sleep_client, fake_vllm):
    """Spec section 15.2: a burst against a sleeping engine produces
    exactly one wake operation and every request succeeds."""
    await _sleep_engine(sleep_broker, fake_vllm)
    fake_vllm.wake_gate = asyncio.Event()

    async def request():
        return await sleep_client.post(
            "/v1/chat/completions", json={"model": "fake-model", "messages": []}
        )

    tasks = [asyncio.create_task(request()) for _ in range(8)]
    # Wait until the wake attempt is actually in flight, then release it.
    while fake_vllm.wake_calls == 0:  # noqa: ASYNC110 - no hook to await
        await asyncio.sleep(0.01)
    fake_vllm.wake_gate.set()

    responses = await asyncio.gather(*tasks)
    assert [r.status_code for r in responses] == [200] * 8
    assert fake_vllm.wake_calls == 1
    assert fake_vllm.violations == 0


async def test_wake_failure_returns_503_and_marks_error(sleep_broker, sleep_client, fake_vllm):
    await _sleep_engine(sleep_broker, fake_vllm)
    fake_vllm.wake_fail = True

    resp = await sleep_client.post(
        "/v1/chat/completions", json={"model": "fake-model", "messages": []}
    )

    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "engine_unavailable"
    assert resp.headers["retry-after"] == "60"
    assert sleep_broker.engine.state == EngineState.ERROR
    assert fake_vllm.violations == 0

    # While in ERROR, requests keep failing fast without new wake attempts.
    wake_calls_before = fake_vllm.wake_calls
    resp2 = await sleep_client.post(
        "/v1/chat/completions", json={"model": "fake-model", "messages": []}
    )
    assert resp2.status_code == 503
    assert fake_vllm.wake_calls == wake_calls_before


async def test_admin_wake_recovers_from_error(sleep_broker, sleep_client, admin_client, fake_vllm):
    await _sleep_engine(sleep_broker, fake_vllm)
    fake_vllm.wake_fail = True
    await sleep_client.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert sleep_broker.engine.state == EngineState.ERROR

    fake_vllm.wake_fail = False
    resp = await admin_client.post("/internal/v1/engines/main/wake")
    assert resp.status_code == 202
    assert resp.json()["state"] == "awake"

    resp = await sleep_client.post(
        "/v1/chat/completions", json={"model": "fake-model", "messages": []}
    )
    assert resp.status_code == 200
    assert fake_vllm.violations == 0


async def test_wake_timeout_is_a_failure(sleep_broker, sleep_client, fake_vllm):
    """/wake_up returns OK but the engine keeps reporting sleeping —
    the GB10 silent-EngineCore-death mode (vllm#50011). The poll must
    time out, mark ERROR, and never forward the request."""
    await _sleep_engine(sleep_broker, fake_vllm)
    fake_vllm.wake_noop = True
    sleep_broker.engine.cfg.wake_timeout_seconds = 0.05

    resp = await sleep_client.post(
        "/v1/chat/completions", json={"model": "fake-model", "messages": []}
    )

    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "engine_unavailable"
    assert sleep_broker.engine.state == EngineState.ERROR
    assert fake_vllm.violations == 0


async def test_refresh_state_reconstructs_from_engine(fake_vllm):
    from conftest import make_broker

    fake_vllm.sleeping = True
    broker = make_broker(fake_vllm, sleep_enabled=True)
    assert broker.engine.state == EngineState.UNKNOWN
    await broker.engine.refresh_state()
    assert broker.engine.state == EngineState.SLEEPING

    fake_vllm.sleeping = False
    await broker.engine.refresh_state()
    assert broker.engine.state == EngineState.AWAKE
    await broker.client.aclose()
