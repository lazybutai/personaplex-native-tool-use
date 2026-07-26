"""Policy-free live V2 action dispatch and executor-observation scheduling.

The neural model remains the only component that chooses calls, controls, and
spoken behavior.  This module authenticates completed parsed actions, tracks
hidden REF leases, and places externally authored observations on the next
legal five-slot input frames.  It intentionally contains no planner.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Callable
import uuid

from .action_v2 import (
    DEFAULT_MANIFEST,
    MICRO_TOKENS_PER_FRAME,
    PAD_TOKEN_ID,
    ActionKind,
    ActionParser,
    ActionV2Manifest,
    MicroPosition,
    ParsedAction,
    RefLease,
    RefLeaseError,
    RefLeaseGuard,
    TerminalLeaseReceipt,
)
from .environment_v2 import (
    EnvironmentEventV2,
    EnvironmentProtocolError,
    ScheduledEnvironmentEvent,
    validate_environment_fragments,
)


class LiveRuntimeV2Error(ValueError):
    """Raised for an invalid live clock, dispatch, or executor transition."""


@dataclass(frozen=True, slots=True)
class LiveEnvironmentFrame:
    """One executor-authored input frame ready for `AgentMicroLMGen.step`."""

    frame: int
    ids: tuple[int, ...]
    mask: tuple[bool, ...]
    completed_events: tuple[ScheduledEnvironmentEvent, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.frame, bool) or not isinstance(self.frame, int) or self.frame < 0:
            raise LiveRuntimeV2Error("live environment frame index must be non-negative")
        ids = tuple(self.ids)
        mask = tuple(self.mask)
        completed = tuple(self.completed_events)
        if len(ids) != MICRO_TOKENS_PER_FRAME or len(mask) != MICRO_TOKENS_PER_FRAME:
            raise LiveRuntimeV2Error("live environment frame must contain exactly five slots")
        padding_started = False
        for token, present in zip(ids, mask, strict=True):
            if not isinstance(present, bool):
                raise LiveRuntimeV2Error("live environment mask values must be booleans")
            if token == PAD_TOKEN_ID:
                padding_started = True
                if present:
                    raise LiveRuntimeV2Error("PAD=-1 cannot be visible environment context")
            elif (
                isinstance(token, bool)
                or not isinstance(token, int)
                or token < 0
                or token >= 256
            ):
                raise LiveRuntimeV2Error("live environment token is outside 0..255")
            elif not present:
                raise LiveRuntimeV2Error("every live environment token requires a true mask")
            elif padding_started:
                raise LiveRuntimeV2Error("live environment PAD must be a frame suffix")
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "completed_events", completed)

    @property
    def has_observation(self) -> bool:
        return any(self.mask)


@dataclass(slots=True)
class _ActiveSerialization:
    event: EnvironmentEventV2
    symbolic_fragments: tuple[tuple[str, ...], ...]
    fragments: tuple[tuple[int, ...], ...]
    fragment_index: int = 0
    injected_start: MicroPosition | None = None
    injected_end: MicroPosition | None = None


class LiveEnvironmentFifo:
    """Incremental equivalent of the offline FIFO scheduler on an 80 ms clock."""

    def __init__(self, manifest: ActionV2Manifest = DEFAULT_MANIFEST) -> None:
        self.manifest = manifest
        self._heap: list[tuple[int, int, str, EnvironmentEventV2]] = []
        self._event_ids: set[str] = set()
        self._event_sequences: set[int] = set()
        self._terminal_leases: set[tuple[int, str, int]] = set()
        self._active: _ActiveSerialization | None = None
        self._next_frame = 0
        self._last_ack_frame = -1
        self._completed: list[ScheduledEnvironmentEvent] = []
        self._pending_receipts: list[TerminalLeaseReceipt] = []

    @property
    def next_frame(self) -> int:
        return self._next_frame

    @property
    def queued_event_count(self) -> int:
        return len(self._heap) + (1 if self._active is not None else 0)

    def submit(
        self,
        event: EnvironmentEventV2,
        *,
        lease_guard: RefLeaseGuard,
    ) -> None:
        """Authenticate and queue one external event without making it visible early."""

        if not isinstance(event, EnvironmentEventV2):
            raise LiveRuntimeV2Error("live FIFO accepts only EnvironmentEventV2 values")
        if not isinstance(lease_guard, RefLeaseGuard):
            raise LiveRuntimeV2Error("live FIFO submission requires a RefLeaseGuard")
        if event.manifest.fingerprint != self.manifest.fingerprint:
            raise LiveRuntimeV2Error("live event uses a different V2 manifest")
        if event.event_id in self._event_ids:
            raise LiveRuntimeV2Error(f"duplicate live event_id {event.event_id!r}")
        if event.event_seq in self._event_sequences:
            raise LiveRuntimeV2Error(f"duplicate live event_seq {event.event_seq}")
        lease = lease_guard.validate_event(
            event.ref,
            call_uuid=event.call_uuid,
            lease=event.lease,
        )
        if event.issued_frame != lease.issued_frame:
            raise RefLeaseError(
                "live event issued_frame must match the authenticated CALL_END frame"
            )
        identity = (lease.ref, lease.call_uuid, lease.lease)
        if identity in self._terminal_leases:
            raise LiveRuntimeV2Error("an event cannot be queued after its lease's terminal event")
        if event.is_terminal:
            self._terminal_leases.add(identity)

        self._event_ids.add(event.event_id)
        self._event_sequences.add(event.event_seq)
        heapq.heappush(
            self._heap,
            (event.available_frame, event.event_seq, event.event_id, event),
        )

    def poll(self, frame: int) -> LiveEnvironmentFrame:
        """Serialize everything causally available into exactly one five-slot frame."""

        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise LiveRuntimeV2Error("live poll frame must be a non-negative integer")
        if frame != self._next_frame:
            raise LiveRuntimeV2Error(
                f"live environment clock expected frame {self._next_frame}, got {frame}"
            )
        ids = [PAD_TOKEN_ID] * MICRO_TOKENS_PER_FRAME
        mask = [False] * MICRO_TOKENS_PER_FRAME
        cursor = 0
        completed_this_frame: list[ScheduledEnvironmentEvent] = []

        while cursor < MICRO_TOKENS_PER_FRAME:
            if self._active is None:
                if not self._heap or self._heap[0][0] > frame:
                    break
                _, _, _, event = heapq.heappop(self._heap)
                symbolic = validate_environment_fragments(
                    event,
                    event.symbolic_fragments(),
                    manifest=self.manifest,
                )
                fragments = tuple(
                    tuple(self.manifest.environment_id(token) for token in fragment)
                    for fragment in symbolic
                )
                self._active = _ActiveSerialization(event, symbolic, fragments)

            active = self._active
            assert active is not None
            fragment = active.fragments[active.fragment_index]
            if len(fragment) > MICRO_TOKENS_PER_FRAME - cursor:
                break
            start = MicroPosition(frame, cursor)
            if active.injected_start is None:
                active.injected_start = start
            for offset, token in enumerate(fragment):
                ids[cursor + offset] = token
                mask[cursor + offset] = True
            cursor += len(fragment)
            active.injected_end = MicroPosition(frame, cursor - 1)
            active.fragment_index += 1

            if active.fragment_index == len(active.fragments):
                assert active.injected_start is not None and active.injected_end is not None
                scheduled = ScheduledEnvironmentEvent(
                    event=active.event,
                    symbolic_fragments=active.symbolic_fragments,
                    fragments=active.fragments,
                    injected_start=active.injected_start,
                    injected_end=active.injected_end,
                    visible_frame=active.injected_end.frame + 1,
                )
                completed_this_frame.append(scheduled)
                self._completed.append(scheduled)
                receipt = scheduled.terminal_receipt
                if receipt is not None:
                    self._pending_receipts.append(receipt)
                self._active = None

        result = LiveEnvironmentFrame(
            frame=frame,
            ids=tuple(ids),
            mask=tuple(mask),
            completed_events=tuple(completed_this_frame),
        )
        self._next_frame += 1
        return result

    def release_visible_refs(
        self,
        lease_guard: RefLeaseGuard,
        *,
        current_frame: int,
    ) -> tuple[RefLease, ...]:
        """Acknowledge model visibility and release each terminal receipt once."""

        if isinstance(current_frame, bool) or not isinstance(current_frame, int) or current_frame < 0:
            raise LiveRuntimeV2Error("visibility frame must be a non-negative integer")
        if current_frame < self._last_ack_frame:
            raise LiveRuntimeV2Error("visibility acknowledgements must be monotonic")
        released: list[RefLease] = []
        still_pending: list[TerminalLeaseReceipt] = []
        for receipt in self._pending_receipts:
            if receipt.visible_frame <= current_frame:
                released.append(lease_guard.release_terminal(receipt))
            else:
                still_pending.append(receipt)
        self._pending_receipts = still_pending
        self._last_ack_frame = current_frame
        return tuple(released)

    def drain_completed(self) -> tuple[ScheduledEnvironmentEvent, ...]:
        completed = tuple(self._completed)
        self._completed.clear()
        return completed


@dataclass(frozen=True, slots=True)
class ActionDispatchV2:
    """Policy-free runtime outcome for one completed model action."""

    action: ParsedAction
    lease: RefLease | None
    executor_required: bool

    def __post_init__(self) -> None:
        if self.action.kind is ActionKind.CALL:
            if self.lease is None or not self.executor_required:
                raise LiveRuntimeV2Error("CALL dispatch requires a new authenticated lease")
        elif self.action.kind is ActionKind.CANCEL:
            if self.lease is None or not self.executor_required:
                raise LiveRuntimeV2Error("CANCEL dispatch requires its active lease")
        elif self.lease is not None or self.executor_required:
            raise LiveRuntimeV2Error("NOOP/control actions must not invoke an executor")


class DuplexToolRuntimeV2:
    """Session-scoped bridge between sampled model actions and trusted events."""

    def __init__(
        self,
        manifest: ActionV2Manifest = DEFAULT_MANIFEST,
        *,
        uuid_factory: Callable[[], str] | None = None,
    ) -> None:
        self.manifest = manifest
        self.parser = ActionParser(manifest)
        self.leases = RefLeaseGuard(manifest)
        self.environment = LiveEnvironmentFifo(manifest)
        self._uuid_factory = uuid_factory or (lambda: str(uuid.uuid4()))
        self._call_uuids: set[str] = set()

    @property
    def active_refs(self) -> frozenset[int]:
        return self.leases.active_refs

    def dispatch_action(
        self,
        action: ParsedAction,
        *,
        call_uuid: str | None = None,
    ) -> ActionDispatchV2:
        if not isinstance(action, ParsedAction):
            raise LiveRuntimeV2Error("runtime dispatch requires a parsed V2 action")
        validated = self.parser.parse(action.packet, active_refs=self.leases)
        if validated != action:
            raise LiveRuntimeV2Error("parsed action metadata does not match its grammar packet")
        if action.kind is ActionKind.CALL:
            identity = call_uuid if call_uuid is not None else self._uuid_factory()
            if not isinstance(identity, str) or not identity:
                raise LiveRuntimeV2Error("live call UUID factory returned an invalid identity")
            if identity in self._call_uuids:
                raise LiveRuntimeV2Error("live call UUIDs must be unique within a session")
            lease = self.leases.reserve_parsed(action, call_uuid=identity)
            self._call_uuids.add(identity)
            return ActionDispatchV2(action, lease, True)
        if action.kind is ActionKind.CANCEL:
            if action.ref is None:
                raise LiveRuntimeV2Error("parsed cancellation is missing its REF")
            return ActionDispatchV2(action, self.leases.lease_for(action.ref), True)
        return ActionDispatchV2(action, None, False)

    def submit_event(self, event: EnvironmentEventV2) -> None:
        self.environment.submit(event, lease_guard=self.leases)

    def environment_frame(self, frame: int) -> LiveEnvironmentFrame:
        return self.environment.poll(frame)

    def acknowledge_model_frame(self, frame: int) -> tuple[RefLease, ...]:
        return self.environment.release_visible_refs(self.leases, current_frame=frame)


__all__ = [
    "ActionDispatchV2",
    "DuplexToolRuntimeV2",
    "LiveEnvironmentFifo",
    "LiveEnvironmentFrame",
    "LiveRuntimeV2Error",
]
