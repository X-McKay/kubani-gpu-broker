"""Exclusive GPU lease arbitration (spec sections 10, 11, 22).

Ownership transitions:

    AVAILABLE --acquire--> DRAINING --engines quiet--> TRAINING
    TRAINING --release/expiry-with-no-jobs--> AVAILABLE

Fail-closed rules baked in:
- a lease is never granted unless the state is AVAILABLE;
- an expired heartbeat alone never recovers the GPU — only expiry with
  zero matching Jobs does (spec section 11);
- Kubernetes API uncertainty during reconstruction lands in RECOVERING,
  which refuses grants and refuses wakes but lets already-awake engines
  keep serving (spec section 22.4).

Reclaim modes (kubani amendment 3):
- "sleep": the broker sleeps every sleep-capable engine before granting
  (~80 GiB freed on GB10; LoRA-class jobs).
- "restart": the broker drains and grants without sleeping; the training
  workflow owns stopping/starting the engines (the broker deliberately
  has no pod-mutation RBAC, spec sections 22.2 and 24.4).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import timedelta

from .engines import Engine
from .kube import KubeClient, LeaseRecord, utcnow
from .state import GpuOwnership, GpuOwnershipState


class LeaseError(Exception):
    pass


class LeaseConflictError(LeaseError):
    pass


class DrainTimeoutError(LeaseError):
    pass


class LeaseNotFoundError(LeaseError):
    pass


class JobsStillActiveError(LeaseError):
    pass


class LeaseUnavailableError(LeaseError):
    """Lease subsystem disabled or in RECOVERING."""


class ActiveLease:
    def __init__(self, lease_id: str, owner: str, workload_id: str, reclaim: str) -> None:
        self.lease_id = lease_id
        self.owner = owner
        self.workload_id = workload_id
        self.reclaim = reclaim
        self.renewed_at = utcnow()

    @property
    def holder(self) -> str:
        return f"{self.owner}:{self.workload_id}"


class LeaseManager:
    def __init__(
        self,
        ownership: GpuOwnership,
        engines: dict[str, Engine],
        kube: KubeClient | None,
        duration_seconds: int = 180,
        drain_timeout_seconds: float = 300.0,
        drain_poll_interval_seconds: float = 0.5,
    ) -> None:
        self._ownership = ownership
        self._engines = engines
        self._kube = kube
        self._duration = duration_seconds
        self._drain_timeout = drain_timeout_seconds
        self._drain_poll = drain_poll_interval_seconds
        self._lock = asyncio.Lock()
        self.active: ActiveLease | None = None

    @property
    def enabled(self) -> bool:
        return self._kube is not None

    # -- startup reconstruction (spec section 22.1) --------------------------

    async def reconstruct(self) -> None:
        if self._kube is None:
            return
        try:
            record = await self._kube.read_lease()
            if record is None or not record.holder or record.holder == "inference":
                self._ownership.state = GpuOwnershipState.AVAILABLE
                return
            lease_id = record.lease_id or ""
            jobs = await self._kube.active_jobs_for_lease(lease_id) if lease_id else 0
            expired = self._record_expired(record)
            if jobs > 0 or not expired:
                owner, _, workload = record.holder.partition(":")
                self.active = ActiveLease(
                    lease_id or "unknown", owner, workload, record.reclaim or "sleep"
                )
                self._ownership.state = GpuOwnershipState.TRAINING
            else:
                await self._kube.clear_lease()
                self._ownership.state = GpuOwnershipState.AVAILABLE
        except Exception:
            # Uncertainty fails closed (spec section 22.4).
            self._ownership.state = GpuOwnershipState.RECOVERING

    @staticmethod
    def _record_expired(record: LeaseRecord) -> bool:
        if record.renew_time is None or record.duration_seconds <= 0:
            return True
        return utcnow() > record.renew_time + timedelta(seconds=record.duration_seconds)

    # -- lease lifecycle -----------------------------------------------------

    async def acquire(
        self,
        owner: str,
        workload_id: str,
        reclaim: str = "sleep",
        drain_timeout_seconds: float | None = None,
    ) -> ActiveLease:
        if self._kube is None:
            raise LeaseUnavailableError("lease subsystem is not enabled")
        async with self._lock:
            # Idempotency: the workflow ID is the key (spec section 20).
            if (
                self.active is not None
                and self.active.owner == owner
                and self.active.workload_id == workload_id
            ):
                return self.active
            if self._ownership.state == GpuOwnershipState.RECOVERING:
                raise LeaseUnavailableError("GPU ownership is RECOVERING; fix state first")
            if self._ownership.state != GpuOwnershipState.AVAILABLE:
                raise LeaseConflictError(
                    f"GPU is {self._ownership.state.value}"
                    + (f" (held by {self.active.holder})" if self.active else "")
                )

            self._ownership.state = GpuOwnershipState.DRAINING
            try:
                await self._drain(drain_timeout_seconds or self._drain_timeout)
                if reclaim == "sleep":
                    for engine in self._engines.values():
                        if engine.cfg.sleep_enabled:
                            await engine.sleep(manual=True)

                lease = ActiveLease(uuid.uuid4().hex, owner, workload_id, reclaim)
                await self._kube.write_lease(
                    LeaseRecord(
                        holder=lease.holder,
                        lease_id=lease.lease_id,
                        workload_id=workload_id,
                        reclaim=reclaim,
                        acquire_time=utcnow(),
                        renew_time=utcnow(),
                        duration_seconds=self._duration,
                    )
                )
            except Exception:
                # No lease was granted; give the GPU back to inference.
                self._ownership.state = GpuOwnershipState.AVAILABLE
                raise

            self.active = lease
            self._ownership.state = GpuOwnershipState.TRAINING
            return lease

    async def _drain(self, drain_timeout: float) -> None:
        deadline = time.monotonic() + drain_timeout
        while any(e.in_flight > 0 for e in self._engines.values()):
            if time.monotonic() >= deadline:
                raise DrainTimeoutError(f"inference did not drain within {drain_timeout:.0f}s")
            await asyncio.sleep(self._drain_poll)

    async def renew(self, lease_id: str) -> ActiveLease:
        if self._kube is None:
            raise LeaseUnavailableError("lease subsystem is not enabled")
        async with self._lock:
            if self.active is None or self.active.lease_id != lease_id:
                raise LeaseNotFoundError(lease_id)
            self.active.renewed_at = utcnow()
            record = await self._kube.read_lease()
            if record is not None:
                record.renew_time = utcnow()
                await self._kube.write_lease(record)
            return self.active

    async def release(self, lease_id: str) -> None:
        if self._kube is None:
            raise LeaseUnavailableError("lease subsystem is not enabled")
        async with self._lock:
            if self.active is None or self.active.lease_id != lease_id:
                raise LeaseNotFoundError(lease_id)
            jobs = await self._kube.active_jobs_for_lease(lease_id)
            if jobs > 0:
                raise JobsStillActiveError(f"{jobs} Job(s) still active for lease {lease_id}")
            await self._kube.clear_lease()
            self.active = None
            # Engines stay asleep; the next inference request wakes them
            # (spec section 10.3).
            self._ownership.state = GpuOwnershipState.AVAILABLE

    # -- expiry watchdog (spec section 11) -----------------------------------

    async def tick(self) -> None:
        """Periodic check: recover an expired lease only when no Job
        carrying its ID remains active."""
        if self._kube is None or self._ownership.state != GpuOwnershipState.TRAINING:
            return
        async with self._lock:
            if self.active is None:
                return
            age = (utcnow() - self.active.renewed_at).total_seconds()
            if age <= self._duration:
                return
            try:
                jobs = await self._kube.active_jobs_for_lease(self.active.lease_id)
            except Exception:
                return  # uncertainty: remain TRAINING (fail closed)
            if jobs > 0:
                return
            await self._kube.clear_lease()
            self.active = None
            self._ownership.state = GpuOwnershipState.AVAILABLE

    def snapshot(self) -> dict:
        return {
            "enabled": self.enabled,
            "gpu_owner": self._ownership.state.value,
            "lease": (
                {
                    "lease_id": self.active.lease_id,
                    "holder": self.active.holder,
                    "reclaim": self.active.reclaim,
                    "renewed_at": self.active.renewed_at.isoformat(),
                    "duration_seconds": self._duration,
                }
                if self.active
                else None
            ),
        }
