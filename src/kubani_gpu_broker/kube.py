"""Kubernetes access for lease persistence and job introspection.

The broker's RBAC (spec section 24.4): read/write one Lease, list Jobs.
The protocol keeps the lease logic testable without a cluster; the real
implementation is deliberately thin.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol


@dataclass
class LeaseRecord:
    holder: str
    lease_id: str | None
    workload_id: str | None
    reclaim: str | None
    acquire_time: datetime | None
    renew_time: datetime | None
    duration_seconds: int


class KubeClient(Protocol):
    async def read_lease(self) -> LeaseRecord | None: ...

    async def write_lease(self, record: LeaseRecord) -> None: ...

    async def clear_lease(self) -> None: ...

    async def active_jobs_for_lease(self, lease_id: str) -> int: ...


def utcnow() -> datetime:
    return datetime.now(UTC)


class InClusterKubeClient:
    """Real client. Imports kubernetes_asyncio lazily so tests and
    non-cluster environments never need it."""

    _ANNOTATION_PREFIX = "kubani.ai/"

    def __init__(self, lease_name: str, namespace: str) -> None:
        self._lease_name = lease_name
        self._namespace = namespace
        self._api = None

    async def _coordination(self):
        from kubernetes_asyncio import client, config

        if self._api is None:
            config.load_incluster_config()
            self._api = client.ApiClient()
        return client.CoordinationV1Api(self._api)

    async def _batch(self):
        from kubernetes_asyncio import client

        await self._coordination()  # ensures config + shared ApiClient
        return client.BatchV1Api(self._api)

    async def read_lease(self) -> LeaseRecord | None:
        from kubernetes_asyncio.client.exceptions import ApiException

        api = await self._coordination()
        try:
            lease = await api.read_namespaced_lease(self._lease_name, self._namespace)
        except ApiException as exc:
            if exc.status == 404:
                return None
            raise
        spec = lease.spec
        annotations = lease.metadata.annotations or {}
        return LeaseRecord(
            holder=spec.holder_identity or "inference",
            lease_id=annotations.get(f"{self._ANNOTATION_PREFIX}lease-id"),
            workload_id=annotations.get(f"{self._ANNOTATION_PREFIX}workload-id"),
            reclaim=annotations.get(f"{self._ANNOTATION_PREFIX}reclaim"),
            acquire_time=spec.acquire_time,
            renew_time=spec.renew_time,
            duration_seconds=spec.lease_duration_seconds or 0,
        )

    async def write_lease(self, record: LeaseRecord) -> None:
        from kubernetes_asyncio import client
        from kubernetes_asyncio.client.exceptions import ApiException

        api = await self._coordination()
        annotations = {
            f"{self._ANNOTATION_PREFIX}lease-id": record.lease_id or "",
            f"{self._ANNOTATION_PREFIX}workload-id": record.workload_id or "",
            f"{self._ANNOTATION_PREFIX}reclaim": record.reclaim or "",
        }
        body = client.V1Lease(
            metadata=client.V1ObjectMeta(
                name=self._lease_name, namespace=self._namespace, annotations=annotations
            ),
            spec=client.V1LeaseSpec(
                holder_identity=record.holder,
                acquire_time=record.acquire_time,
                renew_time=record.renew_time,
                lease_duration_seconds=record.duration_seconds,
            ),
        )
        try:
            await api.replace_namespaced_lease(self._lease_name, self._namespace, body)
        except ApiException as exc:
            if exc.status == 404:
                await api.create_namespaced_lease(self._namespace, body)
            else:
                raise

    async def clear_lease(self) -> None:
        await self.write_lease(
            LeaseRecord(
                holder="inference",
                lease_id=None,
                workload_id=None,
                reclaim=None,
                acquire_time=None,
                renew_time=utcnow(),
                duration_seconds=0,
            )
        )

    async def active_jobs_for_lease(self, lease_id: str) -> int:
        api = await self._batch()
        jobs = await api.list_job_for_all_namespaces(
            label_selector=f"kubani.ai/gpu-lease-id={lease_id}"
        )
        active = 0
        for job in jobs.items:
            if (job.status.active or 0) > 0:
                active += 1
        return active
