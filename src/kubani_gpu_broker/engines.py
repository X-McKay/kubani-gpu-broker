"""Engine: request accounting, vLLM sleep driver, and lifecycle transitions.

All state transitions happen under a per-engine transition lock, and every
predicate is re-checked after acquiring it (spec section 19). Concurrent
requests that find the engine sleeping queue on that lock, so a burst
produces exactly one /wake_up call — the single-flight guarantee.

Wake verification never trusts vLLM's /health endpoint: it stays green
after a native EngineCore death (kubani spec amendment 5). Instead the
wake path polls /is_sleeping and then performs an active probe
(GET /v1/models must return 200).
"""

from __future__ import annotations

import asyncio
import time

import httpx

from .config import EngineConfig, PoliciesConfig
from .state import (
    EngineBusyError,
    EngineState,
    EngineUnavailableError,
    GpuOwnership,
    GpuUnavailableError,
    WakeFailedError,
)
from .telemetry import Metrics


class Engine:
    def __init__(
        self,
        name: str,
        cfg: EngineConfig,
        policies: PoliciesConfig,
        client: httpx.AsyncClient,
        metrics: Metrics,
        ownership: GpuOwnership,
    ) -> None:
        self.name = name
        self.cfg = cfg
        self.policies = policies
        self.base_url = cfg.base_url.rstrip("/")
        self._client = client
        self._metrics = metrics
        self._ownership = ownership

        self._in_flight = 0
        self.last_activity = time.monotonic()
        self.last_wake = time.monotonic()
        self.transition_lock = asyncio.Lock()
        # Without sleep support the engine is definitionally awake; with it,
        # the true state is unknown until refresh_state() or the first
        # ensure_awake() asks the engine.
        self._state = EngineState.AWAKE if not cfg.sleep_enabled else EngineState.UNKNOWN
        self._metrics.set_engine_state(name, self._state)

    # -- request accounting -------------------------------------------------

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def acquire(self) -> None:
        self._in_flight += 1

    def release(self) -> None:
        self._in_flight -= 1
        self.last_activity = time.monotonic()

    def idle_seconds(self) -> float:
        if self._in_flight > 0:
            return 0.0
        return time.monotonic() - self.last_activity

    # -- state --------------------------------------------------------------

    @property
    def state(self) -> EngineState:
        return self._state

    def _set_state(self, state: EngineState) -> None:
        self._state = state
        self._metrics.set_engine_state(self.name, state)

    async def refresh_state(self) -> None:
        """Reconstruct state from the engine itself (spec section 22.1)."""
        if not self.cfg.sleep_enabled:
            return
        async with self.transition_lock:
            if self._state in (EngineState.WAKING, EngineState.ERROR):
                return
            sleeping = await self._is_sleeping()
            self._set_state(EngineState.SLEEPING if sleeping else EngineState.AWAKE)

    # -- vLLM sleep driver (dev-mode endpoints; never exposed publicly) -----

    # Control-plane calls carry their own timeouts: the proxy client is
    # deliberately unbounded for streaming, but a hung /sleep or /wake_up
    # must fail the transition, not hang the caller.

    def _control_timeout(self) -> httpx.Timeout:
        return httpx.Timeout(self.cfg.wake_timeout_seconds)

    async def _is_sleeping(self) -> bool:
        resp = await self._client.get(
            f"{self.base_url}/is_sleeping", timeout=self._control_timeout()
        )
        resp.raise_for_status()
        return bool(resp.json()["is_sleeping"])

    async def _post_sleep(self, level: int) -> None:
        resp = await self._client.post(
            f"{self.base_url}/sleep", params={"level": level}, timeout=self._control_timeout()
        )
        resp.raise_for_status()

    async def _post_wake_up(self) -> None:
        resp = await self._client.post(
            f"{self.base_url}/wake_up", timeout=self._control_timeout()
        )
        resp.raise_for_status()

    async def _verify_serving(self) -> None:
        """Active probe: the engine must actually answer an API request."""
        resp = await self._client.get(
            f"{self.base_url}/v1/models", timeout=self._control_timeout()
        )
        resp.raise_for_status()

    # -- transitions --------------------------------------------------------

    async def ensure_awake(self) -> None:
        """Wake the engine if needed; single-flight across callers.

        Raises GpuUnavailableError when an exclusive workload owns the GPU,
        EngineUnavailableError when the engine is in ERROR, and
        WakeFailedError when this call's wake attempt fails.
        """
        if not self.cfg.sleep_enabled:
            return
        if not self._ownership.available():
            raise GpuUnavailableError(self.name)
        if self._state == EngineState.AWAKE:
            return

        async with self.transition_lock:
            # Re-check everything after acquiring the lock (spec section 19).
            if self._state == EngineState.AWAKE:
                return
            if not self._ownership.available():
                raise GpuUnavailableError(self.name)
            if self._state == EngineState.ERROR:
                raise EngineUnavailableError(self.name)

            if not await self._is_sleeping():
                self._set_state(EngineState.AWAKE)
                return

            self._set_state(EngineState.WAKING)
            started = time.monotonic()
            try:
                await self._post_wake_up()
                await self._poll_until_awake()
                await self._verify_serving()
            except Exception as exc:
                self._set_state(EngineState.ERROR)
                self._metrics.wake_total.labels(engine=self.name, result="failure").inc()
                raise WakeFailedError(f"{self.name}: {exc}") from exc

            self._set_state(EngineState.AWAKE)
            self.last_wake = time.monotonic()
            self._metrics.wake_total.labels(engine=self.name, result="success").inc()
            self._metrics.wake_duration_seconds.labels(engine=self.name).observe(
                time.monotonic() - started
            )

    async def _poll_until_awake(self) -> None:
        deadline = time.monotonic() + self.cfg.wake_timeout_seconds
        while await self._is_sleeping():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"engine {self.name} still sleeping after wake_up")
            await asyncio.sleep(self.cfg.wake_poll_interval_seconds)

    async def sleep(self, level: int | None = None, *, manual: bool = False) -> None:
        """Put the engine to sleep. Manual (admin) sleeps refuse while
        requests are in flight; idle sleeps re-verify all predicates under
        the lock (spec section 9.3)."""
        if not self.cfg.sleep_enabled:
            raise EngineUnavailableError(f"{self.name}: sleep not enabled")
        resolved_level = level if level is not None else self.cfg.idle_sleep_level

        async with self.transition_lock:
            if self._state == EngineState.SLEEPING:
                return
            if self._in_flight > 0:
                raise EngineBusyError(f"{self.name}: {self._in_flight} requests in flight")
            if not manual and not self._should_idle_sleep_locked():
                return

            try:
                await self._post_sleep(resolved_level)
                if not await self._is_sleeping():
                    raise RuntimeError("engine did not report sleeping after /sleep")
            except Exception as exc:
                self._metrics.sleep_total.labels(
                    engine=self.name, level=str(resolved_level), result="failure"
                ).inc()
                raise WakeFailedError(f"{self.name}: sleep failed: {exc}") from exc

            self._set_state(EngineState.SLEEPING)
            self._metrics.sleep_total.labels(
                engine=self.name, level=str(resolved_level), result="success"
            ).inc()

    async def reset_error(self) -> None:
        """Admin escape hatch: leave ERROR so a wake can be retried."""
        async with self.transition_lock:
            if self._state == EngineState.ERROR:
                self._set_state(EngineState.UNKNOWN)

    # -- idle policy (spec section 9.3) -------------------------------------

    def should_idle_sleep(self) -> bool:
        return (
            self.cfg.sleep_enabled
            and self._ownership.available()
            and self._state == EngineState.AWAKE
            and self._in_flight == 0
            and self.idle_seconds() >= self.cfg.idle_timeout_seconds
            and (time.monotonic() - self.last_wake) >= self.policies.min_awake_seconds
        )

    def _should_idle_sleep_locked(self) -> bool:
        # Same predicates, re-evaluated while holding the transition lock.
        return self.should_idle_sleep()

    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "state": self._state.value,
            "in_flight": self._in_flight,
            "idle_seconds": round(self.idle_seconds(), 3),
            "sleep_enabled": self.cfg.sleep_enabled,
            "base_url": self.base_url,
        }
