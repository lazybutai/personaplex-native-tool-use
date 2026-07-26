"""Typed contract between the duplex model and the tool executor.

The model may emit an :class:`ActionCall`.  Tool outcomes are observations from
the environment, never model targets.  A later PersonaPlex integration will
serialize these events into dedicated result tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import TypeAlias


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

_CALL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_.-]{1,63}$")


class ToolStatus(str, Enum):
    """Executor outcomes that the model can observe."""

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ObservationKind(str, Enum):
    """Event type injected into the model's causal observation stream."""

    PENDING = "pending"
    RESULT = "result"


@dataclass(frozen=True, slots=True)
class ActionCall:
    """A complete, grammar-valid action emitted by the model.

    ``issued_frame`` is the 80 ms frame containing the final grammar-valid
    action token (the logical ``CALL_END``), not the start of a packet.  The
    bridge makes no policy decision: it either validates and dispatches this
    completed call or rejects it.
    """

    call_id: str
    tool_name: str
    arguments: dict[str, JSONValue]
    issued_frame: int

    def __post_init__(self) -> None:
        if not _CALL_ID.fullmatch(self.call_id):
            raise ValueError(f"invalid call_id: {self.call_id!r}")
        if not _TOOL_NAME.fullmatch(self.tool_name):
            raise ValueError(f"invalid tool_name: {self.tool_name!r}")
        if self.issued_frame < 0:
            raise ValueError("issued_frame must be non-negative")


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """A deterministic result returned by a registered tool handler."""

    status: ToolStatus
    payload: dict[str, JSONValue]
    delay_frames: int = 1

    def __post_init__(self) -> None:
        if self.delay_frames < 1:
            raise ValueError("delay_frames must be at least one causal frame")


@dataclass(frozen=True, slots=True)
class ObservationEvent:
    """A source-typed event ready for model-context injection."""

    kind: ObservationKind
    call_id: str
    tool_name: str
    due_frame: int
    status: ToolStatus | None = None
    payload: dict[str, JSONValue] | None = None

    def __post_init__(self) -> None:
        if not _CALL_ID.fullmatch(self.call_id):
            raise ValueError(f"invalid call_id: {self.call_id!r}")
        if not _TOOL_NAME.fullmatch(self.tool_name):
            raise ValueError(f"invalid tool_name: {self.tool_name!r}")
        if self.due_frame < 0:
            raise ValueError("due_frame must be non-negative")
        if self.kind is ObservationKind.PENDING and self.status is not None:
            raise ValueError("pending events cannot carry a terminal status")
        if self.kind is ObservationKind.RESULT and self.status is None:
            raise ValueError("result events require a terminal status")
