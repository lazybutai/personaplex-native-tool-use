"""Temporary in-band serialization for the first PersonaPlex state experiment.

The explicit action/result lanes come later.  This serializer lets us prove
causal result injection using PersonaPlex's existing forced agent-text-token
hook while keeping the transport contract typed and auditable.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
import json

from .protocol import ObservationEvent, ObservationKind


def render_observation(event: ObservationEvent) -> str:
    """Render a deterministic, compact training/injection representation."""

    if event.kind is ObservationKind.PENDING:
        return f"<ENV_PENDING id={event.call_id} tool={event.tool_name}>"

    payload = json.dumps(event.payload or {}, separators=(",", ":"), sort_keys=True)
    return (
        f"<ENV_RESULT id={event.call_id} tool={event.tool_name} "
        f"status={event.status.value}>{payload}</ENV_RESULT>"
    )


class InBandTokenQueue:
    """Temporary one-token-per-frame observation injector.

    PersonaPlex's current generator accepts one forced agent-text token per
    80 ms frame.  This queue preserves causal ordering for the compatibility
    experiment.  The future explicit result lane will replace it.
    """

    def __init__(self) -> None:
        self._tokens: deque[int] = deque()

    def enqueue(self, event: ObservationEvent, encode: Callable[[str], list[int]]) -> int:
        token_ids = encode(render_observation(event))
        if not token_ids:
            raise ValueError("observation encoding must contain at least one token")
        self._tokens.extend(token_ids)
        return len(token_ids)

    def pop_next(self) -> int | None:
        """Return one token for the next model frame, or no forced token."""

        return self._tokens.popleft() if self._tokens else None

    @property
    def pending_tokens(self) -> int:
        return len(self._tokens)
