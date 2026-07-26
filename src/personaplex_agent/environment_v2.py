"""Causal V2 executor-observation lane for PersonaPlex-Agent.

Environment tokens are trusted input-only context.  They are never action
targets and they never contain a model-visible UUID.  The scheduler converts
source-typed executor events into bounded five-micro-slot observation
fragments, preserving FIFO order and recording the exact causal injection
positions needed by the action REF lease guard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping

from .action_v2 import (
    DEFAULT_MANIFEST,
    MAX_REFS,
    MICRO_TOKENS_PER_FRAME,
    PAD_TOKEN_ID,
    ActionV2Manifest,
    MicroPosition,
    RefLease,
    RefLeaseError,
    RefLeaseGuard,
    TerminalLeaseReceipt,
)


class EnvironmentProtocolError(ValueError):
    """Base error for an invalid executor event or scheduled observation."""


class EnvironmentFragmentError(EnvironmentProtocolError):
    """Raised when symbolic OBS_BEGIN/OBS_MORE/OBS_END grammar is invalid."""


class EnvironmentKind(str, Enum):
    """Fixed status tokens the model can observe from the environment."""

    PENDING = "PENDING"
    OK = "OK"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    STALE = "STALE"


_ERROR_PAYLOAD_TOKENS = frozenset(
    {"ERR_REF_BUSY", "ERR_SCHEMA", "ERR_PERMISSION", "ERR_EXECUTOR"}
)


def _normalise_ref(ref: int | str, *, max_refs: int = MAX_REFS) -> int:
    if isinstance(ref, bool):
        raise EnvironmentProtocolError("environment REF must be an integer index or REF_n token")
    if isinstance(ref, int):
        index = ref
    elif isinstance(ref, str) and ref.startswith("REF_"):
        try:
            index = int(ref.removeprefix("REF_"))
        except ValueError as exc:
            raise EnvironmentProtocolError(f"invalid environment REF token: {ref!r}") from exc
    else:
        raise EnvironmentProtocolError("environment REF must be an integer index or REF_n token")
    if index < 0 or index >= max_refs:
        raise EnvironmentProtocolError(f"REF_{index} is outside the V2 0..{max_refs - 1} range")
    return index


def _position_key(position: MicroPosition) -> tuple[int, int]:
    return (position.frame, position.micro)


@dataclass(frozen=True, slots=True)
class EnvironmentEventV2:
    """An executor-authored observation that becomes available on a future tick.

    ``call_uuid`` and ``lease`` remain runtime metadata.  The model sees only
    the fixed symbolic ``REF_n`` token, tool/status token, and bounded payload
    vocabulary produced by :meth:`symbolic_fragments`.
    """

    event_id: str
    event_seq: int
    available_frame: int
    issued_frame: int
    ref: int | str
    tool_name: str
    kind: EnvironmentKind | str
    payload_tokens: tuple[str, ...] = ()
    call_uuid: str = ""
    lease: int = 0
    source: str = "tool_executor"
    manifest: ActionV2Manifest = field(default=DEFAULT_MANIFEST, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id:
            raise EnvironmentProtocolError("environment event_id is required")
        if isinstance(self.event_seq, bool) or not isinstance(self.event_seq, int) or self.event_seq < 0:
            raise EnvironmentProtocolError("environment event_seq must be a non-negative integer")
        if (
            isinstance(self.issued_frame, bool)
            or not isinstance(self.issued_frame, int)
            or self.issued_frame < 0
        ):
            raise EnvironmentProtocolError("environment issued_frame must be non-negative")
        if (
            isinstance(self.available_frame, bool)
            or not isinstance(self.available_frame, int)
            or self.available_frame < self.issued_frame + 1
        ):
            raise EnvironmentProtocolError(
                "environment event available_frame must be at least issued_frame + 1"
            )
        if not isinstance(self.manifest, ActionV2Manifest):
            raise EnvironmentProtocolError("environment event requires an ActionV2Manifest")
        ref = _normalise_ref(self.ref, max_refs=self.manifest.max_refs)
        try:
            kind = self.kind if isinstance(self.kind, EnvironmentKind) else EnvironmentKind(self.kind)
        except ValueError as exc:
            raise EnvironmentProtocolError(f"unknown environment event kind: {self.kind!r}") from exc
        if not isinstance(self.tool_name, str) or not self.tool_name:
            raise EnvironmentProtocolError("environment tool_name is required")
        try:
            tool = self.manifest.tool(self.tool_name)
        except ValueError as exc:
            raise EnvironmentProtocolError(str(exc)) from exc
        if not isinstance(self.call_uuid, str) or not self.call_uuid:
            raise EnvironmentProtocolError("environment event requires the hidden call_uuid")
        if isinstance(self.lease, bool) or not isinstance(self.lease, int) or self.lease < 1:
            raise EnvironmentProtocolError("environment event lease must be a positive integer")
        if self.source != "tool_executor":
            raise EnvironmentProtocolError("only tool_executor may author V2 environment events")

        payload = tuple(self.payload_tokens)
        if any(not isinstance(token, str) or not token for token in payload):
            raise EnvironmentProtocolError("environment payloads must use symbolic token names")
        if kind is EnvironmentKind.PENDING:
            if payload:
                raise EnvironmentProtocolError("PENDING observations cannot carry payload tokens")
        elif kind is EnvironmentKind.OK:
            if any(token not in tool.result_tokens for token in payload):
                raise EnvironmentProtocolError(
                    f"OK payload for {tool.name!r} is outside its fixed result vocabulary"
                )
        elif any(token not in _ERROR_PAYLOAD_TOKENS for token in payload):
            raise EnvironmentProtocolError(
                "non-OK terminal payloads may use only fixed ERR_* environment tokens"
            )

        object.__setattr__(self, "ref", ref)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "payload_tokens", payload)

    @property
    def is_terminal(self) -> bool:
        return self.kind is not EnvironmentKind.PENDING

    @property
    def ref_token(self) -> str:
        return f"REF_{self.ref}"

    @property
    def environment_tool_token(self) -> str:
        return self.manifest.tool(self.tool_name).environment_token

    def symbolic_fragments(self) -> tuple[tuple[str, ...], ...]:
        """Serialize one event without exposing hidden UUID/lease metadata.

        A no-payload observation occupies one full five-token header frame:
        ``OBS_BEGIN KIND REF TOOL OBS_END``.  Payload-bearing observations use
        a full first header fragment followed by ``OBS_MORE REF ... OBS_END``
        fragments, so only the final fragment closes the observation.
        """

        header = ["OBS_BEGIN", self.kind.value, self.ref_token, self.environment_tool_token]
        payload = list(self.payload_tokens)
        if not payload:
            return (tuple((*header, "OBS_END")),)

        fragments: list[tuple[str, ...]] = [tuple((*header, payload.pop(0)))]
        continuation = ("OBS_MORE", self.ref_token)
        while len(payload) > 2:
            fragments.append(tuple((*continuation, *payload[:3])))
            del payload[:3]
        fragments.append(tuple((*continuation, *payload, "OBS_END")))
        return tuple(fragments)

    def to_dict(self) -> dict[str, object]:
        """Metadata snapshot; token serialization itself omits UUID/lease."""

        return {
            "event_id": self.event_id,
            "event_seq": self.event_seq,
            "available_frame": self.available_frame,
            "issued_frame": self.issued_frame,
            "ref": self.ref,
            "tool_name": self.tool_name,
            "kind": self.kind.value,
            "payload_tokens": list(self.payload_tokens),
            "call_uuid": self.call_uuid,
            "lease": self.lease,
            "source": self.source,
            "manifest_fingerprint": self.manifest.fingerprint,
        }


def validate_environment_fragments(
    event: EnvironmentEventV2,
    fragments: Iterable[Iterable[str]],
    *,
    manifest: ActionV2Manifest = DEFAULT_MANIFEST,
) -> tuple[tuple[str, ...], ...]:
    """Validate the complete OBS_BEGIN / OBS_MORE / OBS_END fragment FSA."""

    if event.manifest.fingerprint != manifest.fingerprint:
        raise EnvironmentFragmentError("event and fragment validator use different V2 manifests")
    normalized = tuple(tuple(fragment) for fragment in fragments)
    if not normalized:
        raise EnvironmentFragmentError("environment observation requires at least one fragment")
    if any(not fragment or len(fragment) > MICRO_TOKENS_PER_FRAME for fragment in normalized):
        raise EnvironmentFragmentError("each environment fragment must contain one to five tokens")
    first = normalized[0]
    required_header = ("OBS_BEGIN", event.kind.value, event.ref_token, event.environment_tool_token)
    if first[:4] != required_header:
        raise EnvironmentFragmentError("first environment fragment must begin OBS_BEGIN KIND REF TOOL")

    for index, fragment in enumerate(normalized):
        is_final = index == len(normalized) - 1
        if index:
            if len(fragment) < 2 or fragment[:2] != ("OBS_MORE", event.ref_token):
                raise EnvironmentFragmentError(
                    "continuation fragments must begin OBS_MORE with the same REF token"
                )
            if "OBS_BEGIN" in fragment:
                raise EnvironmentFragmentError("OBS_BEGIN may appear only in the first fragment")
        if is_final:
            if fragment[-1] != "OBS_END" or fragment.count("OBS_END") != 1:
                raise EnvironmentFragmentError("only the final environment fragment may end with OBS_END")
        elif "OBS_END" in fragment:
            raise EnvironmentFragmentError("OBS_END may appear only in the final environment fragment")

    expected = event.symbolic_fragments()
    if normalized != expected:
        raise EnvironmentFragmentError(
            "environment fragments do not exactly match the event's fixed symbolic payload"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class EnvironmentMicroLane:
    """Input-only [T, 5] executor IDs and an explicit presence mask."""

    ids: tuple[tuple[int, ...], ...]
    mask: tuple[tuple[bool, ...], ...]
    padding_id: int = PAD_TOKEN_ID

    def __post_init__(self) -> None:
        if self.padding_id != PAD_TOKEN_ID:
            raise EnvironmentProtocolError("V2 environment padding is fixed at PAD=-1")
        ids = tuple(tuple(frame) for frame in self.ids)
        mask = tuple(tuple(frame) for frame in self.mask)
        if not ids:
            raise EnvironmentProtocolError("environment lane requires at least one 80 ms frame")
        if len(ids) != len(mask):
            raise EnvironmentProtocolError("environment ids and mask must have equal frame counts")
        for frame_ids, frame_mask in zip(ids, mask, strict=True):
            if len(frame_ids) != MICRO_TOKENS_PER_FRAME or len(frame_mask) != MICRO_TOKENS_PER_FRAME:
                raise EnvironmentProtocolError("every environment frame has exactly five micro-slots")
            padding_started = False
            for token_id, present in zip(frame_ids, frame_mask, strict=True):
                if not isinstance(present, bool):
                    raise EnvironmentProtocolError("environment lane mask values must be booleans")
                if token_id == PAD_TOKEN_ID:
                    padding_started = True
                    if present:
                        raise EnvironmentProtocolError("PAD=-1 is absent context, never an environment token")
                elif (
                    isinstance(token_id, bool)
                    or not isinstance(token_id, int)
                    or token_id < 0
                    or token_id >= 256
                ):
                    raise EnvironmentProtocolError(
                        "environment lane IDs must be in the fixed 0..255 range"
                    )
                elif not present:
                    raise EnvironmentProtocolError("every environment token must have a true presence mask")
                elif padding_started:
                    raise EnvironmentProtocolError("environment PAD must be a contiguous frame suffix")
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "mask", mask)

    @property
    def frame_count(self) -> int:
        return len(self.ids)

    @property
    def environment_ids(self) -> tuple[tuple[int, ...], ...]:
        return self.ids

    @property
    def environment_mask(self) -> tuple[tuple[bool, ...], ...]:
        return self.mask

    def visible_tokens(self, frame: int) -> tuple[int, ...]:
        return tuple(
            token_id
            for token_id, present in zip(self.ids[frame], self.mask[frame], strict=True)
            if present
        )

    def to_dict(self) -> dict[str, object]:
        return {"ids": [list(frame) for frame in self.ids], "mask": [list(frame) for frame in self.mask]}


@dataclass(frozen=True, slots=True)
class ScheduledEnvironmentEvent:
    """One event's exact fragments and causal positions after FIFO scheduling."""

    event: EnvironmentEventV2
    symbolic_fragments: tuple[tuple[str, ...], ...]
    fragments: tuple[tuple[int, ...], ...]
    injected_start: MicroPosition
    injected_end: MicroPosition
    visible_frame: int

    def __post_init__(self) -> None:
        symbolic = tuple(tuple(fragment) for fragment in self.symbolic_fragments)
        fragments = tuple(tuple(fragment) for fragment in self.fragments)
        if len(symbolic) != len(fragments) or not fragments:
            raise EnvironmentProtocolError("scheduled environment event requires matching non-empty fragments")
        try:
            validate_environment_fragments(self.event, symbolic, manifest=self.event.manifest)
        except EnvironmentFragmentError as exc:
            raise EnvironmentProtocolError("scheduled environment fragments fail the V2 observation grammar") from exc
        expected_ids = tuple(
            tuple(self.event.manifest.environment_id(token) for token in fragment)
            for fragment in symbolic
        )
        if fragments != expected_ids:
            raise EnvironmentProtocolError(
                "scheduled environment IDs do not match the fixed symbolic observation fragments"
            )
        if self.visible_frame != self.injected_end.frame + 1:
            raise EnvironmentProtocolError("terminal/context visibility is exactly injected_end.frame + 1")
        if _position_key(self.injected_start) > _position_key(self.injected_end):
            raise EnvironmentProtocolError("environment injected_start must not follow injected_end")
        if self.injected_start.frame < self.event.available_frame:
            raise EnvironmentProtocolError("environment event was injected before its available frame")
        object.__setattr__(self, "symbolic_fragments", symbolic)
        object.__setattr__(self, "fragments", fragments)

    @property
    def is_terminal(self) -> bool:
        return self.event.is_terminal

    @property
    def terminal_receipt(self) -> TerminalLeaseReceipt | None:
        if not self.event.is_terminal:
            return None
        return TerminalLeaseReceipt(
            ref=self.event.ref,
            call_uuid=self.event.call_uuid,
            lease=self.event.lease,
            available_frame=self.event.available_frame,
            injected_end=self.injected_end,
            source=self.event.source,
            terminal_token="OBS_END",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "event": self.event.to_dict(),
            "symbolic_fragments": [list(fragment) for fragment in self.symbolic_fragments],
            "fragments": [list(fragment) for fragment in self.fragments],
            "injected_start": {"frame": self.injected_start.frame, "micro": self.injected_start.micro},
            "injected_end": {"frame": self.injected_end.frame, "micro": self.injected_end.micro},
            "visible_frame": self.visible_frame,
        }


@dataclass(frozen=True, slots=True)
class EnvironmentSchedule:
    """Immutable FIFO result lane plus scheduled event metadata."""

    lane: EnvironmentMicroLane
    scheduled_events: tuple[ScheduledEnvironmentEvent, ...]
    manifest_fingerprint: str

    def __post_init__(self) -> None:
        events = tuple(self.scheduled_events)
        if not isinstance(self.manifest_fingerprint, str) or not self.manifest_fingerprint:
            raise EnvironmentProtocolError("environment schedule requires a manifest fingerprint")
        object.__setattr__(self, "scheduled_events", events)

    @property
    def ids(self) -> tuple[tuple[int, ...], ...]:
        return self.lane.ids

    @property
    def mask(self) -> tuple[tuple[bool, ...], ...]:
        return self.lane.mask

    @property
    def frame_count(self) -> int:
        return self.lane.frame_count

    def event(self, event_id: str) -> ScheduledEnvironmentEvent:
        for scheduled in self.scheduled_events:
            if scheduled.event.event_id == event_id:
                return scheduled
        raise KeyError(event_id)

    @property
    def terminal_receipts(self) -> tuple[TerminalLeaseReceipt, ...]:
        """Scheduler-authenticated terminal receipts, ordered by FIFO injection."""

        return tuple(
            receipt
            for scheduled in self.scheduled_events
            if (receipt := scheduled.terminal_receipt) is not None
        )

    def release_visible_refs(
        self,
        lease_guard: RefLeaseGuard,
        *,
        current_frame: int,
    ) -> tuple[RefLease, ...]:
        """Release only terminal refs that are causally visible *now*.

        Building an offline schedule must not mutate a live guard early: an
        ``OBS_END`` injected in frame ``t`` can affect model output only in
        frame ``t + 1``.  The runtime calls this method as its clock advances.
        Releasing the same receipt twice is intentionally rejected by the
        guard as a stale lifecycle operation.
        """

        if not isinstance(lease_guard, RefLeaseGuard):
            raise EnvironmentProtocolError("release_visible_refs requires a RefLeaseGuard")
        if isinstance(current_frame, bool) or not isinstance(current_frame, int) or current_frame < 0:
            raise EnvironmentProtocolError("current_frame must be a non-negative integer")
        released: list[RefLease] = []
        for receipt in self.terminal_receipts:
            if receipt.visible_frame <= current_frame:
                released.append(lease_guard.release_terminal(receipt))
        return tuple(released)

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_fingerprint": self.manifest_fingerprint,
            "lane": self.lane.to_dict(),
            "scheduled_events": [event.to_dict() for event in self.scheduled_events],
        }


class EnvironmentFifoScheduler:
    """Schedule executor events deterministically into five-slot input frames."""

    def __init__(self, manifest: ActionV2Manifest = DEFAULT_MANIFEST) -> None:
        self.manifest = manifest

    def schedule(
        self,
        frame_count: int,
        events: Iterable[EnvironmentEventV2],
        *,
        lease_guard: RefLeaseGuard | None = None,
    ) -> EnvironmentSchedule:
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
            raise EnvironmentProtocolError("environment frame_count must be a positive integer")
        event_values = tuple(events)
        if not all(isinstance(event, EnvironmentEventV2) for event in event_values):
            raise EnvironmentProtocolError("environment scheduler accepts only EnvironmentEventV2 values")
        for event in event_values:
            if event.manifest.fingerprint != self.manifest.fingerprint:
                raise EnvironmentProtocolError("environment event manifest does not match this scheduler")
        event_ids = [event.event_id for event in event_values]
        event_seqs = [event.event_seq for event in event_values]
        if len(set(event_ids)) != len(event_ids):
            raise EnvironmentProtocolError("environment event_id values must be unique")
        if len(set(event_seqs)) != len(event_seqs):
            raise EnvironmentProtocolError("environment event_seq values must be unique for FIFO ordering")

        ordered_events = tuple(sorted(event_values, key=lambda event: (event.available_frame, event.event_seq)))
        ids = [[PAD_TOKEN_ID] * MICRO_TOKENS_PER_FRAME for _ in range(frame_count)]
        mask = [[False] * MICRO_TOKENS_PER_FRAME for _ in range(frame_count)]
        cursor_frame = 0
        cursor_micro = 0
        scheduled: list[ScheduledEnvironmentEvent] = []
        terminal_leases: set[tuple[int, str, int]] = set()

        for event in ordered_events:
            lease_identity = (event.ref, event.call_uuid, event.lease)
            if lease_identity in terminal_leases:
                raise EnvironmentProtocolError(
                    "environment event follows a terminal OBS_END for the same REF lease"
                )
            if lease_guard is not None:
                current_lease = lease_guard.validate_event(
                    event.ref,
                    call_uuid=event.call_uuid,
                    lease=event.lease,
                )
                if event.issued_frame != current_lease.issued_frame:
                    raise RefLeaseError(
                        "environment event issued_frame must match the authenticated CALL_END frame"
                    )
                if event.available_frame < current_lease.issued_frame + 1:
                    raise RefLeaseError(
                        "environment event cannot be available before the authenticated CALL_END + 1"
                    )

            symbolic_fragments = validate_environment_fragments(
                event,
                event.symbolic_fragments(),
                manifest=self.manifest,
            )
            encoded_fragments = tuple(
                tuple(self.manifest.environment_id(token) for token in fragment)
                for fragment in symbolic_fragments
            )

            injected_start: MicroPosition | None = None
            injected_end: MicroPosition | None = None
            for fragment in encoded_fragments:
                if cursor_frame < event.available_frame:
                    cursor_frame = event.available_frame
                    cursor_micro = 0
                if cursor_micro + len(fragment) > MICRO_TOKENS_PER_FRAME:
                    cursor_frame += 1
                    cursor_micro = 0
                if cursor_frame >= frame_count:
                    raise EnvironmentProtocolError(
                        "environment FIFO schedule exceeds the available aligned timeline"
                    )
                if cursor_frame < event.available_frame:
                    raise EnvironmentProtocolError("environment scheduler attempted pre-availability injection")
                start = MicroPosition(cursor_frame, cursor_micro)
                for offset, token_id in enumerate(fragment):
                    micro = cursor_micro + offset
                    if ids[cursor_frame][micro] != PAD_TOKEN_ID:
                        raise EnvironmentProtocolError("environment scheduler attempted to overwrite a micro-slot")
                    ids[cursor_frame][micro] = token_id
                    mask[cursor_frame][micro] = True
                end = MicroPosition(cursor_frame, cursor_micro + len(fragment) - 1)
                if injected_start is None:
                    injected_start = start
                injected_end = end
                cursor_micro += len(fragment)
                if cursor_micro == MICRO_TOKENS_PER_FRAME:
                    cursor_frame += 1
                    cursor_micro = 0

            if injected_start is None or injected_end is None:  # defensive; fragments are non-empty
                raise EnvironmentProtocolError("environment event produced no injectible tokens")
            scheduled_event = ScheduledEnvironmentEvent(
                event=event,
                symbolic_fragments=symbolic_fragments,
                fragments=encoded_fragments,
                injected_start=injected_start,
                injected_end=injected_end,
                visible_frame=injected_end.frame + 1,
            )
            scheduled.append(scheduled_event)
            receipt = scheduled_event.terminal_receipt
            if receipt is not None:
                terminal_leases.add(lease_identity)

        lane = EnvironmentMicroLane(
            ids=tuple(tuple(frame) for frame in ids),
            mask=tuple(tuple(frame) for frame in mask),
        )
        return EnvironmentSchedule(
            lane=lane,
            scheduled_events=tuple(scheduled),
            manifest_fingerprint=self.manifest.fingerprint,
        )


# Naming aliases retain a concise API at call sites without duplicating logic.
EnvironmentEvent = EnvironmentEventV2
EnvironmentScheduler = EnvironmentFifoScheduler
EnvironmentSchedulerV2 = EnvironmentFifoScheduler
ExecutorEventV2 = EnvironmentEventV2


__all__ = [
    "EnvironmentEvent",
    "EnvironmentEventV2",
    "EnvironmentFifoScheduler",
    "EnvironmentFragmentError",
    "EnvironmentKind",
    "EnvironmentMicroLane",
    "EnvironmentProtocolError",
    "EnvironmentSchedule",
    "EnvironmentScheduler",
    "EnvironmentSchedulerV2",
    "ExecutorEventV2",
    "ScheduledEnvironmentEvent",
    "validate_environment_fragments",
]
