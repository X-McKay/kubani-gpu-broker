"""Engine lifecycle states, GPU ownership, and broker errors.

Spec sections 7.1 and 7.2. GPU ownership is a placeholder until the
exclusive-lease phase: it always reports AVAILABLE, but the wake and
idle paths already consult it, so the lease work only has to change
this object, not the call sites.
"""

from __future__ import annotations

from enum import StrEnum


class EngineState(StrEnum):
    UNKNOWN = "unknown"
    AWAKE = "awake"
    WAKING = "waking"
    SLEEPING = "sleeping"
    ERROR = "error"


# Stable numeric encoding for the engine_state gauge.
ENGINE_STATE_VALUES: dict[EngineState, int] = {
    EngineState.UNKNOWN: 0,
    EngineState.AWAKE: 1,
    EngineState.WAKING: 2,
    EngineState.SLEEPING: 3,
    EngineState.ERROR: 4,
}


class GpuOwnershipState(StrEnum):
    AVAILABLE = "AVAILABLE"
    DRAINING = "DRAINING"
    TRAINING = "TRAINING"
    RECOVERING = "RECOVERING"


class GpuOwnership:
    """Physical GPU ownership. Fail closed: anything but AVAILABLE
    forbids waking inference (spec section 4.4)."""

    def __init__(self) -> None:
        self.state = GpuOwnershipState.AVAILABLE

    def available(self) -> bool:
        return self.state == GpuOwnershipState.AVAILABLE


class BrokerError(Exception):
    pass


class GpuUnavailableError(BrokerError):
    """The GPU is owned by an exclusive workload; do not wake inference."""


class WakeFailedError(BrokerError):
    """A wake transition failed; the engine is now in ERROR."""


class EngineUnavailableError(BrokerError):
    """The engine is in ERROR state; requests must not be forwarded."""


class EngineBusyError(BrokerError):
    """A manual sleep was requested while requests are in flight."""
