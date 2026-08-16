"""Idle monitor (spec section 9).

`tick()` is separated from the timing loop so tests can drive it
deterministically. The loop only runs when auto_sleep_enabled is true —
the qualification gate (spec section 14) keeps it off by default.
"""

from __future__ import annotations

import asyncio
import logging

from .engines import Engine
from .state import EngineBusyError

logger = logging.getLogger(__name__)


class IdleLoop:
    def __init__(self, engines: dict[str, Engine], check_interval_seconds: float) -> None:
        self._engines = engines
        self._interval = check_interval_seconds
        self._task: asyncio.Task | None = None

    async def tick(self) -> None:
        for engine in self._engines.values():
            if not engine.should_idle_sleep():
                continue
            try:
                # Predicates are re-checked under the transition lock; a
                # request that slipped in makes this a no-op.
                await engine.sleep(manual=False)
            except EngineBusyError:
                pass
            except Exception:
                logger.exception("idle sleep failed for engine %s", engine.name)

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            await self.tick()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
