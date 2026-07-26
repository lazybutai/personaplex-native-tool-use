"""A policy-free, causal tool-result bridge for PersonaPlex-Agent.

This module is intentionally model-agnostic.  PersonaPlex will decide whether
to create an action; the bridge validates it, executes a registered safe tool,
and queues typed observations for a later model tick.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import TypeAlias

from .protocol import ActionCall, JSONValue, ObservationEvent, ObservationKind, ToolOutcome, ToolStatus


ToolHandler: TypeAlias = Callable[[dict[str, JSONValue]], Awaitable[ToolOutcome]]


class ToolRegistry:
    """Explicit allow-list of executor functions."""

    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, tool_name: str, handler: ToolHandler) -> None:
        if tool_name in self._handlers:
            raise ValueError(f"tool already registered: {tool_name}")
        self._handlers[tool_name] = handler

    def handler_for(self, tool_name: str) -> ToolHandler | None:
        return self._handlers.get(tool_name)


class CausalToolBridge:
    """Queue executor observations without allowing retrospective injection.

    An outcome can only be observed at a later frame than the action that caused
    it.  This mirrors the causal constraint required by the eventual model
    training interleaver and prevents accidental future-result leakage.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._seen_call_ids: set[str] = set()
        self._events: dict[int, list[ObservationEvent]] = defaultdict(list)
        self._last_drained_frame = -1

    async def submit(self, call: ActionCall) -> tuple[ObservationEvent, ObservationEvent]:
        """Validate, execute, and schedule pending plus terminal observations."""

        if call.call_id in self._seen_call_ids:
            raise ValueError(f"duplicate call_id: {call.call_id}")
        self._seen_call_ids.add(call.call_id)

        pending = ObservationEvent(
            kind=ObservationKind.PENDING,
            call_id=call.call_id,
            tool_name=call.tool_name,
            due_frame=call.issued_frame + 1,
        )
        handler = self._registry.handler_for(call.tool_name)
        if handler is None:
            outcome = ToolOutcome(
                status=ToolStatus.ERROR,
                payload={"reason": "tool_not_allowed"},
                delay_frames=1,
            )
        else:
            try:
                outcome = await handler(call.arguments)
            except TimeoutError:
                outcome = ToolOutcome(
                    status=ToolStatus.TIMEOUT,
                    payload={"reason": "executor_timeout"},
                    delay_frames=1,
                )
            except Exception as error:  # Tool failures must be observed, never hidden.
                outcome = ToolOutcome(
                    status=ToolStatus.ERROR,
                    payload={"reason": "executor_error", "type": type(error).__name__},
                    delay_frames=1,
                )

        # The current v1 environment lane accepts one logical observation per
        # frame.  Keep a pending acknowledgement and its terminal outcome on
        # distinct causal frames even for an immediate executor response.
        result_due_frame = max(
            call.issued_frame + outcome.delay_frames,
            pending.due_frame + 1,
        )
        result = ObservationEvent(
            kind=ObservationKind.RESULT,
            call_id=call.call_id,
            tool_name=call.tool_name,
            due_frame=result_due_frame,
            status=outcome.status,
            payload=outcome.payload,
        )
        self._events[pending.due_frame].append(pending)
        self._events[result.due_frame].append(result)
        return pending, result

    def drain_ready(self, frame: int) -> tuple[ObservationEvent, ...]:
        """Return source-typed events whose causal due frame has arrived."""

        if frame < self._last_drained_frame:
            raise ValueError("cannot move the causal clock backward")
        ready: list[ObservationEvent] = []
        for due_frame in sorted(key for key in self._events if key <= frame):
            ready.extend(self._events.pop(due_frame))
        self._last_drained_frame = frame
        return tuple(ready)
