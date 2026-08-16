"""Engine registry and per-engine request accounting.

The idle-sleep state machine (spec section 7.2) arrives in a later phase;
this module currently tracks what that phase will need: an accurate
in-flight count and a last-activity timestamp that only advances when a
request fully completes (spec section 9.2 — streaming requests count as
active until the stream ends or the client disconnects).
"""

from __future__ import annotations

import time


class Engine:
    def __init__(self, name: str, base_url: str) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self._in_flight = 0
        self.last_activity = time.monotonic()

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
