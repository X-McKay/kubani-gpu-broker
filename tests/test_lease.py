"""Exclusive GPU lease arbitration: drain, grant, block, release, expiry,
reconstruction. The fake KubeClient is the in-memory stand-in for the
coordination.k8s.io Lease and the Jobs listing."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import httpx
import pytest

from conftest import ADMIN_TOKEN, FakeVllm, make_config
from kubani_gpu_broker.admin import create_admin_app
from kubani_gpu_broker.app import Broker, create_public_app
from kubani_gpu_broker.kube import LeaseRecord, utcnow
from kubani_gpu_broker.state import EngineState, GpuOwnershipState


class FakeKubeClient:
    def __init__(self) -> None:
        self.record: LeaseRecord | None = None
        self.active_jobs: dict[str, int] = {}
        self.fail = False

    async def read_lease(self):
        if self.fail:
            raise RuntimeError("kube api down")
        return self.record

    async def write_lease(self, record: LeaseRecord) -> None:
        if self.fail:
            raise RuntimeError("kube api down")
        self.record = record

    async def clear_lease(self) -> None:
        if self.fail:
            raise RuntimeError("kube api down")
        self.record = LeaseRecord(
            holder="inference",
            lease_id=None,
            workload_id=None,
            reclaim=None,
            acquire_time=None,
            renew_time=utcnow(),
            duration_seconds=0,
        )

    async def active_jobs_for_lease(self, lease_id: str) -> int:
        if self.fail:
            raise RuntimeError("kube api down")
        return self.active_jobs.get(lease_id, 0)


@pytest.fixture
def fake_kube() -> FakeKubeClient:
    return FakeKubeClient()


@pytest.fixture
def lease_broker(fake_vllm: FakeVllm, fake_kube: FakeKubeClient) -> Broker:
    cfg = make_config(
        sleep_enabled=True,
        idle_timeout_seconds=0.05,
        wake_timeout_seconds=2.0,
        wake_poll_interval_seconds=0.01,
    )
    cfg.policies.drain_timeout_seconds = 0.5
    cfg.gpu.leases_enabled = True
    upstream_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake_vllm.app), base_url="http://fake-vllm"
    )
    broker = Broker(cfg, client=upstream_client, kube=fake_kube)
    broker.lease_manager._drain_poll = 0.01
    return broker


@pytest.fixture
async def lease_public(lease_broker: Broker):
    transport = httpx.ASGITransport(app=create_public_app(lease_broker))
    async with httpx.AsyncClient(transport=transport, base_url="http://broker") as client:
        yield client


@pytest.fixture
async def lease_admin(lease_broker: Broker):
    transport = httpx.ASGITransport(app=create_admin_app(lease_broker))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://admin",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    ) as client:
        yield client


LEASE_BODY = {"owner": "fine-tune", "workload_id": "wf-1"}


async def test_acquire_sleeps_engine_and_blocks_inference(
    lease_broker, lease_public, lease_admin, fake_vllm
):
    # Engine serving normally first.
    assert (
        await lease_public.post("/v1/chat/completions", json={"model": "m", "messages": []})
    ).status_code == 200

    resp = await lease_admin.post("/internal/v1/gpu/leases", json=LEASE_BODY)
    assert resp.status_code == 201
    lease = resp.json()
    assert lease["holder"] == "fine-tune:wf-1"

    # Engine slept before the grant; k8s Lease records the holder.
    assert lease_broker.engine.state == EngineState.SLEEPING
    assert fake_vllm.sleeping is True
    assert lease_broker.lease_manager._kube.record.holder == "fine-tune:wf-1"

    # Inference is rejected fast, and no wake is ever attempted.
    wake_calls = fake_vllm.wake_calls
    blocked = await lease_public.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert blocked.status_code == 503
    assert blocked.json()["error"]["type"] == "gpu_temporarily_unavailable"
    assert blocked.headers["retry-after"] == "30"
    assert fake_vllm.wake_calls == wake_calls
    assert fake_vllm.violations == 0


async def test_acquire_waits_for_inflight_stream(
    lease_broker, lease_public, lease_admin, fake_vllm
):
    """Spec section 10.2: existing requests finish before the grant."""
    fake_vllm.stream_gate = asyncio.Event()

    async def consume():
        async with lease_public.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "fake-model", "stream": True, "messages": []},
        ) as resp:
            body = b""
            async for chunk in resp.aiter_raw():
                body += chunk
            return body

    stream_task = asyncio.create_task(consume())
    while lease_broker.engine.in_flight == 0:  # noqa: ASYNC110 - no hook to await
        await asyncio.sleep(0.01)

    acquire_task = asyncio.create_task(lease_admin.post("/internal/v1/gpu/leases", json=LEASE_BODY))
    await asyncio.sleep(0.05)
    assert not acquire_task.done()  # draining: waiting on the stream
    assert lease_broker.ownership.state == GpuOwnershipState.DRAINING

    fake_vllm.stream_gate.set()
    body = await stream_task
    assert b"[DONE]" in body  # the in-flight stream completed intact

    resp = await acquire_task
    assert resp.status_code == 201
    assert lease_broker.ownership.state == GpuOwnershipState.TRAINING


async def test_drain_timeout_returns_gpu_to_inference(
    lease_broker, lease_public, lease_admin, fake_vllm
):
    fake_vllm.stream_gate = asyncio.Event()

    async def consume():
        async with lease_public.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "fake-model", "stream": True, "messages": []},
        ) as resp:
            async for _ in resp.aiter_raw():
                pass

    stream_task = asyncio.create_task(consume())
    while lease_broker.engine.in_flight == 0:  # noqa: ASYNC110 - no hook to await
        await asyncio.sleep(0.01)

    resp = await lease_admin.post(
        "/internal/v1/gpu/leases",
        json={**LEASE_BODY, "drain_timeout_seconds": 0.05},
    )
    assert resp.status_code == 504
    assert lease_broker.ownership.state == GpuOwnershipState.AVAILABLE
    assert lease_broker.lease_manager.active is None

    fake_vllm.stream_gate.set()
    await stream_task


async def test_second_acquire_conflicts_and_idempotent_same_workload(lease_admin, lease_broker):
    first = await lease_admin.post("/internal/v1/gpu/leases", json=LEASE_BODY)
    assert first.status_code == 201

    other = await lease_admin.post(
        "/internal/v1/gpu/leases", json={"owner": "eval", "workload_id": "wf-2"}
    )
    assert other.status_code == 409

    same = await lease_admin.post("/internal/v1/gpu/leases", json=LEASE_BODY)
    assert same.status_code == 201
    assert same.json()["lease_id"] == first.json()["lease_id"]


async def test_release_requires_no_active_jobs_then_frees_gpu(
    lease_admin, lease_public, lease_broker, fake_kube, fake_vllm
):
    lease = (await lease_admin.post("/internal/v1/gpu/leases", json=LEASE_BODY)).json()
    lease_id = lease["lease_id"]

    fake_kube.active_jobs[lease_id] = 1
    resp = await lease_admin.delete(f"/internal/v1/gpu/leases/{lease_id}")
    assert resp.status_code == 409
    assert lease_broker.ownership.state == GpuOwnershipState.TRAINING

    fake_kube.active_jobs[lease_id] = 0
    resp = await lease_admin.delete(f"/internal/v1/gpu/leases/{lease_id}")
    assert resp.status_code == 204
    assert lease_broker.ownership.state == GpuOwnershipState.AVAILABLE

    # Engines stay asleep after release; the next request wakes (spec 10.3).
    assert lease_broker.engine.state == EngineState.SLEEPING
    resp = await lease_public.post(
        "/v1/chat/completions", json={"model": "fake-model", "messages": []}
    )
    assert resp.status_code == 200
    assert fake_vllm.wake_calls == 1
    assert fake_vllm.violations == 0


async def test_restart_reclaim_drains_without_sleeping(lease_admin, lease_broker, fake_vllm):
    resp = await lease_admin.post(
        "/internal/v1/gpu/leases", json={**LEASE_BODY, "reclaim": "restart"}
    )
    assert resp.status_code == 201
    # The broker did not sleep the engine; the workflow owns the restart.
    assert fake_vllm.sleep_calls == 0
    assert lease_broker.ownership.state == GpuOwnershipState.TRAINING


async def test_renew_extends_lease(lease_admin):
    lease = (await lease_admin.post("/internal/v1/gpu/leases", json=LEASE_BODY)).json()
    resp = await lease_admin.post(f"/internal/v1/gpu/leases/{lease['lease_id']}/renew")
    assert resp.status_code == 200
    assert (await lease_admin.post("/internal/v1/gpu/leases/nope/renew")).status_code == 404


async def test_expiry_recovers_only_without_jobs(lease_broker, lease_admin, fake_kube):
    lease = (await lease_admin.post("/internal/v1/gpu/leases", json=LEASE_BODY)).json()
    manager = lease_broker.lease_manager

    # Force expiry.
    manager.active.renewed_at = utcnow() - timedelta(seconds=999)

    fake_kube.active_jobs[lease["lease_id"]] = 1
    await manager.tick()
    assert lease_broker.ownership.state == GpuOwnershipState.TRAINING  # fail closed

    fake_kube.active_jobs[lease["lease_id"]] = 0
    await manager.tick()
    assert lease_broker.ownership.state == GpuOwnershipState.AVAILABLE
    assert manager.active is None


async def test_reconstruct_training_from_lease_and_jobs(fake_vllm, fake_kube):
    fake_kube.record = LeaseRecord(
        holder="fine-tune:wf-9",
        lease_id="abc123",
        workload_id="wf-9",
        reclaim="sleep",
        acquire_time=utcnow(),
        renew_time=utcnow(),
        duration_seconds=180,
    )
    fake_kube.active_jobs["abc123"] = 1

    cfg = make_config(sleep_enabled=True)
    cfg.gpu.leases_enabled = True
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake_vllm.app), base_url="http://fake-vllm"
    )
    broker = Broker(cfg, client=client, kube=fake_kube)
    await broker.lease_manager.reconstruct()
    assert broker.ownership.state == GpuOwnershipState.TRAINING
    assert broker.lease_manager.active.holder == "fine-tune:wf-9"
    await client.aclose()


async def test_reconstruct_clears_stale_lease(fake_vllm, fake_kube):
    fake_kube.record = LeaseRecord(
        holder="fine-tune:wf-9",
        lease_id="abc123",
        workload_id="wf-9",
        reclaim="sleep",
        acquire_time=utcnow() - timedelta(hours=2),
        renew_time=utcnow() - timedelta(hours=1),
        duration_seconds=180,
    )
    cfg = make_config(sleep_enabled=True)
    cfg.gpu.leases_enabled = True
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake_vllm.app), base_url="http://fake-vllm"
    )
    broker = Broker(cfg, client=client, kube=fake_kube)
    await broker.lease_manager.reconstruct()
    assert broker.ownership.state == GpuOwnershipState.AVAILABLE
    assert fake_kube.record.holder == "inference"
    await client.aclose()


async def test_kube_uncertainty_fails_closed(fake_vllm, fake_kube):
    """Spec section 22.4: API failure -> RECOVERING; no grants, no wakes,
    but an already-awake engine keeps serving."""
    fake_kube.fail = True
    cfg = make_config(sleep_enabled=True)
    cfg.gpu.leases_enabled = True
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake_vllm.app), base_url="http://fake-vllm"
    )
    broker = Broker(cfg, client=client, kube=fake_kube)
    await broker.engine.refresh_state()  # engine reachable: AWAKE
    await broker.lease_manager.reconstruct()
    assert broker.ownership.state == GpuOwnershipState.RECOVERING

    transport = httpx.ASGITransport(app=create_public_app(broker))
    async with httpx.AsyncClient(transport=transport, base_url="http://broker") as c:
        # Awake engine still serves (spec 22.4)...
        ok = await c.post("/v1/chat/completions", json={"model": "m", "messages": []})
        assert ok.status_code == 200

        # ...but once sleeping, it is not woken while RECOVERING.
        await broker.engine.sleep(manual=True)
        blocked = await c.post("/v1/chat/completions", json={"model": "m", "messages": []})
        assert blocked.status_code == 503
        assert blocked.json()["error"]["type"] == "gpu_ownership_recovering"
    assert fake_vllm.wake_calls == 0
    assert fake_vllm.violations == 0
    await client.aclose()


async def test_lease_api_disabled_returns_503(admin_client):
    # The default sleep_broker fixture has no kube client.
    resp = await admin_client.post("/internal/v1/gpu/leases", json=LEASE_BODY)
    assert resp.status_code == 503
    assert "not enabled" in resp.json()["detail"]
