"""Deterministic symbolic corpus construction for the PersonaPlex-Agent V2 lanes.

This module is deliberately symbolic.  It builds the native five-slot action
targets and the input-only environment lane while recording the causal proof
that any result-grounded assistant segment begins only after the corresponding
terminal ``OBS_END`` can affect the model.  Base PersonaPlex audio codes are a
typed, unmaterialized ``[17, T]`` contract; audio synthesis/alignment belongs to
a later corpus milestone.

The generator uses SHA-256 domain separation instead of ``random.Random`` so a
scenario is byte-reproducible across Python versions and processes.  Tool calls
remain inside the fixed finite domains in :mod:`personaplex_agent.action_v2`.
No real tool is invoked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping
import uuid

from .action_v2 import (
    ActionCompiler,
    ActionPacketV2,
    ActionParser,
    ActionV2Manifest,
    DEFAULT_MANIFEST,
    MICRO_TOKENS_PER_FRAME,
    MicroLane,
    ParsedAction,
    RefLease,
    RefLeaseGuard,
    materialize_action_lane,
)
from .environment_v2 import (
    EnvironmentEventV2,
    EnvironmentFifoScheduler,
    EnvironmentKind,
    EnvironmentSchedule,
)


CORPUS_V2_VERSION = "ppx-symbolic-corpus-v2.0.0"
BASE_AUDIO_CODEBOOKS = 17
_UUID_NAMESPACE = uuid.UUID("d6c38650-daad-5bd0-909a-980d7a9c2e42")
_TOOLS = ("weather.lookup", "timer.create", "home.set_temperature")


class CorpusV2Error(ValueError):
    """Base error for invalid symbolic V2 corpus data."""


class ReplayValidationError(CorpusV2Error):
    """Raised when serialized or in-memory corpus data fails replay."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_bytes(value: object) -> bytes:
    return _canonical_json(value).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_int(name: str, value: object, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CorpusV2Error(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise CorpusV2Error(f"{name} must be at least {minimum}")
    return value


def _prf(seed: int, domain: str, index: int, *, variant: int = 0) -> int:
    """Stable deterministic integer for one independent finite-domain draw."""

    payload = {
        "corpus_version": CORPUS_V2_VERSION,
        "seed": seed,
        "domain": domain,
        "index": index,
        "variant": variant,
    }
    return int.from_bytes(hashlib.sha256(_canonical_bytes(payload)).digest(), "big")


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """Compact deterministic recipe for one symbolic duplex scenario.

    ``tool_sequence`` and ``terminal_latency_frames`` are optional exact
    overrides useful for curated examples.  Empty tuples select deterministic
    finite-domain values from ``seed``.  ``result_variant`` changes only the
    synthetic executor result, never the model action plan; this is useful for
    causal counterfactual tests.
    """

    seed: int
    frame_count: int = 40
    call_count: int = 3
    first_packet_frame: int = 2
    packet_stride_frames: int = 6
    pending_delay_frames: int = 1
    result_latency_min_frames: int = 3
    result_latency_max_frames: int = 6
    terminal_latency_frames: tuple[int, ...] = ()
    failure_indices: tuple[int, ...] = ()
    tool_sequence: tuple[str, ...] = ()
    grounded_response_delay_frames: int = 0
    result_variant: int = 0

    def __post_init__(self) -> None:
        seed = _require_int("seed", self.seed, minimum=0)
        frame_count = _require_int("frame_count", self.frame_count, minimum=1)
        call_count = _require_int("call_count", self.call_count, minimum=1)
        first = _require_int("first_packet_frame", self.first_packet_frame, minimum=0)
        stride = _require_int("packet_stride_frames", self.packet_stride_frames, minimum=2)
        pending = _require_int("pending_delay_frames", self.pending_delay_frames, minimum=1)
        latency_min = _require_int(
            "result_latency_min_frames", self.result_latency_min_frames, minimum=1
        )
        latency_max = _require_int(
            "result_latency_max_frames", self.result_latency_max_frames, minimum=latency_min
        )
        grounded_delay = _require_int(
            "grounded_response_delay_frames",
            self.grounded_response_delay_frames,
            minimum=0,
        )
        result_variant = _require_int("result_variant", self.result_variant, minimum=0)

        if call_count > DEFAULT_MANIFEST.max_refs:
            raise CorpusV2Error(
                "this symbolic milestone permits at most four simultaneous calls, one per REF"
            )
        if first % 2 or stride % 2:
            raise CorpusV2Error("V2 packet starts and packet stride must be even frame counts")
        if first + (call_count - 1) * stride + 2 > frame_count:
            raise CorpusV2Error("action packets exceed the aligned scenario timeline")
        if latency_min < pending:
            raise CorpusV2Error("terminal latency cannot precede the PENDING event")

        exact_latencies = tuple(self.terminal_latency_frames)
        if exact_latencies and len(exact_latencies) != call_count:
            raise CorpusV2Error(
                "terminal_latency_frames must be empty or contain one latency per call"
            )
        for latency in exact_latencies:
            _require_int("terminal latency", latency, minimum=pending)

        failures = tuple(self.failure_indices)
        if any(isinstance(index, bool) or not isinstance(index, int) for index in failures):
            raise CorpusV2Error("failure_indices must contain integers")
        if len(failures) != len(set(failures)) or any(
            index < 0 or index >= call_count for index in failures
        ):
            raise CorpusV2Error("failure_indices must be unique valid call indices")

        tools = tuple(self.tool_sequence)
        if tools and len(tools) != call_count:
            raise CorpusV2Error("tool_sequence must be empty or contain one tool per call")
        if any(tool not in _TOOLS for tool in tools):
            raise CorpusV2Error("tool_sequence contains a tool outside the fixed V2 catalog")

        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "frame_count", frame_count)
        object.__setattr__(self, "call_count", call_count)
        object.__setattr__(self, "first_packet_frame", first)
        object.__setattr__(self, "packet_stride_frames", stride)
        object.__setattr__(self, "pending_delay_frames", pending)
        object.__setattr__(self, "result_latency_min_frames", latency_min)
        object.__setattr__(self, "result_latency_max_frames", latency_max)
        object.__setattr__(self, "terminal_latency_frames", exact_latencies)
        object.__setattr__(self, "failure_indices", failures)
        object.__setattr__(self, "tool_sequence", tools)
        object.__setattr__(self, "grounded_response_delay_frames", grounded_delay)
        object.__setattr__(self, "result_variant", result_variant)

    @property
    def fingerprint(self) -> str:
        return _sha256(self.to_dict())

    @property
    def scenario_id(self) -> str:
        return f"scenario-{self.fingerprint[:20]}"

    def packet_start_frame(self, call_index: int) -> int:
        self._validate_call_index(call_index)
        return self.first_packet_frame + call_index * self.packet_stride_frames

    def tool_name(self, call_index: int) -> str:
        self._validate_call_index(call_index)
        if self.tool_sequence:
            return self.tool_sequence[call_index]
        return _TOOLS[_prf(self.seed, "call.tool", call_index) % len(_TOOLS)]

    def terminal_latency(self, call_index: int) -> int:
        self._validate_call_index(call_index)
        if self.terminal_latency_frames:
            return self.terminal_latency_frames[call_index]
        width = self.result_latency_max_frames - self.result_latency_min_frames + 1
        return self.result_latency_min_frames + (
            _prf(self.seed, "call.terminal_latency", call_index) % width
        )

    def is_failure(self, call_index: int) -> bool:
        self._validate_call_index(call_index)
        return call_index in self.failure_indices

    def _validate_call_index(self, call_index: int) -> None:
        if (
            isinstance(call_index, bool)
            or not isinstance(call_index, int)
            or call_index < 0
            or call_index >= self.call_count
        ):
            raise CorpusV2Error("call_index is outside this scenario")

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "frame_count": self.frame_count,
            "call_count": self.call_count,
            "first_packet_frame": self.first_packet_frame,
            "packet_stride_frames": self.packet_stride_frames,
            "pending_delay_frames": self.pending_delay_frames,
            "result_latency_min_frames": self.result_latency_min_frames,
            "result_latency_max_frames": self.result_latency_max_frames,
            "terminal_latency_frames": list(self.terminal_latency_frames),
            "failure_indices": list(self.failure_indices),
            "tool_sequence": list(self.tool_sequence),
            "grounded_response_delay_frames": self.grounded_response_delay_frames,
            "result_variant": self.result_variant,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScenarioSpec":
        if not isinstance(data, Mapping):
            raise CorpusV2Error("scenario spec must be an object")
        expected = {
            "seed",
            "frame_count",
            "call_count",
            "first_packet_frame",
            "packet_stride_frames",
            "pending_delay_frames",
            "result_latency_min_frames",
            "result_latency_max_frames",
            "terminal_latency_frames",
            "failure_indices",
            "tool_sequence",
            "grounded_response_delay_frames",
            "result_variant",
        }
        if set(data) != expected:
            raise CorpusV2Error("serialized scenario spec has missing or unknown fields")
        try:
            return cls(
                seed=data["seed"],
                frame_count=data["frame_count"],
                call_count=data["call_count"],
                first_packet_frame=data["first_packet_frame"],
                packet_stride_frames=data["packet_stride_frames"],
                pending_delay_frames=data["pending_delay_frames"],
                result_latency_min_frames=data["result_latency_min_frames"],
                result_latency_max_frames=data["result_latency_max_frames"],
                terminal_latency_frames=tuple(data["terminal_latency_frames"]),
                failure_indices=tuple(data["failure_indices"]),
                tool_sequence=tuple(data["tool_sequence"]),
                grounded_response_delay_frames=data["grounded_response_delay_frames"],
                result_variant=data["result_variant"],
            )
        except (KeyError, TypeError) as exc:
            raise CorpusV2Error("invalid serialized scenario spec") from exc


@dataclass(frozen=True, slots=True)
class BaseAudioCodesContract:
    """Typed placeholder for later PersonaPlex base-code materialization.

    This object never fabricates silence or sentinel codec IDs.  ``codes`` is
    fixed to ``None`` until a real audio-alignment stage supplies exactly 17
    codebooks over this scenario's ``T`` frames.
    """

    frame_count: int
    codebooks: int = BASE_AUDIO_CODEBOOKS
    dtype: str = "int64"
    codes: None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _require_int("base audio frame_count", self.frame_count, minimum=1)
        if self.codebooks != BASE_AUDIO_CODEBOOKS:
            raise CorpusV2Error("PersonaPlex base audio contract is fixed at 17 codebooks")
        if self.dtype != "int64":
            raise CorpusV2Error("base audio code dtype contract is fixed at int64")
        if self.codes is not None:
            raise CorpusV2Error("symbolic corpus V2 must not materialize synthetic base audio codes")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.codebooks, self.frame_count)

    @property
    def materialized(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "materialized": False,
            "codes": None,
            "contract": "PersonaPlex base audio codes [17,T]; deferred to audio alignment",
        }


@dataclass(frozen=True, slots=True)
class SyntheticToolOutcome:
    """Bounded terminal result from the deterministic offline executor."""

    kind: EnvironmentKind
    payload_tokens: tuple[str, ...]
    grounded_text: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EnvironmentKind) or self.kind is EnvironmentKind.PENDING:
            raise CorpusV2Error("synthetic tool outcomes must be terminal environment kinds")
        payload = tuple(self.payload_tokens)
        if any(not isinstance(token, str) or not token for token in payload):
            raise CorpusV2Error("synthetic result payloads must be symbolic token names")
        if not isinstance(self.grounded_text, str) or not self.grounded_text:
            raise CorpusV2Error("synthetic outcomes require a non-empty grounding string")
        object.__setattr__(self, "payload_tokens", payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "payload_tokens": list(self.payload_tokens),
            "grounded_text": self.grounded_text,
        }


class FiniteDomainSyntheticExecutor:
    """Pure deterministic executor for the three fixed V2 tool schemas."""

    def __init__(self, manifest: ActionV2Manifest = DEFAULT_MANIFEST) -> None:
        self.manifest = manifest
        if tuple(tool.name for tool in manifest.tool_schemas) != _TOOLS:
            raise CorpusV2Error("synthetic executor requires the fixed V2 three-tool catalog")

    def execute(
        self,
        action: ParsedAction,
        *,
        spec: ScenarioSpec,
        call_index: int,
    ) -> SyntheticToolOutcome:
        if not action.dispatchable or action.tool_name not in _TOOLS:
            raise CorpusV2Error("synthetic executor accepts only parsed fixed-catalog calls")
        spec._validate_call_index(call_index)
        if spec.is_failure(call_index):
            return SyntheticToolOutcome(
                kind=EnvironmentKind.ERROR,
                payload_tokens=("ERR_EXECUTOR",),
                grounded_text=f"{action.tool_name} ERR_EXECUTOR",
            )

        arguments = dict(action.arguments)
        if action.tool_name == "weather.lookup":
            forecast = _prf(
                spec.seed,
                "result.weather.forecast",
                call_index,
                variant=spec.result_variant,
            ) % 16
            temperature = 10 + (
                _prf(
                    spec.seed,
                    "result.weather.temperature",
                    call_index,
                    variant=spec.result_variant,
                )
                % 31
            )
            payload = (f"FORECAST_{forecast:02d}", f"TEMP_C_{temperature}")
        elif action.tool_name == "timer.create":
            payload = ("ENUM_0",)
        elif action.tool_name == "home.set_temperature":
            payload = ("ENUM_1",)
        else:  # pragma: no cover - guarded above and by the fixed manifest
            raise CorpusV2Error(f"unsupported synthetic tool {action.tool_name!r}")

        argument_text = " ".join(arguments[name] for name in arguments)
        return SyntheticToolOutcome(
            kind=EnvironmentKind.OK,
            payload_tokens=payload,
            grounded_text=" ".join((action.tool_name, argument_text, *payload)).strip(),
        )


@dataclass(frozen=True, slots=True)
class SyntheticCallTrace:
    """Trusted offline metadata joining one action, lease, and executor events."""

    call_index: int
    packet: ActionPacketV2
    parsed_action: ParsedAction
    lease: RefLease
    outcome: SyntheticToolOutcome
    pending_event: EnvironmentEventV2
    terminal_event: EnvironmentEventV2

    def __post_init__(self) -> None:
        _require_int("call trace index", self.call_index, minimum=0)
        if self.parsed_action.packet != self.packet or not self.parsed_action.dispatchable:
            raise CorpusV2Error("call trace requires the matching dispatchable parsed packet")
        if self.parsed_action.ref != self.lease.ref:
            raise CorpusV2Error("call trace action REF does not match its authoritative lease")
        for event in (self.pending_event, self.terminal_event):
            if event.ref != self.lease.ref:
                raise CorpusV2Error("call trace event REF does not match its lease")
            if event.call_uuid != self.lease.call_uuid or event.lease != self.lease.lease:
                raise CorpusV2Error("call trace event hidden identity does not match its lease")
            if event.issued_frame != self.lease.issued_frame:
                raise CorpusV2Error("call trace event issued_frame must come from its lease")
        if self.pending_event.kind is not EnvironmentKind.PENDING:
            raise CorpusV2Error("call trace pending event must use PENDING")
        if not self.terminal_event.is_terminal:
            raise CorpusV2Error("call trace terminal event must be terminal")
        if self.terminal_event.kind is not self.outcome.kind or (
            self.terminal_event.payload_tokens != self.outcome.payload_tokens
        ):
            raise CorpusV2Error("terminal event does not encode its synthetic outcome")

    def to_dict(self) -> dict[str, object]:
        return {
            "call_index": self.call_index,
            "packet": self.packet.to_dict(DEFAULT_MANIFEST),
            "parsed_action": self.parsed_action.to_dict(),
            "runtime_hidden_lease": {
                "ref": self.lease.ref,
                "call_uuid": self.lease.call_uuid,
                "lease": self.lease.lease,
                "issued_frame": self.lease.issued_frame,
                "issued_micro": self.lease.issued_micro,
            },
            "outcome": self.outcome.to_dict(),
            "pending_event": self.pending_event.to_dict(),
            "terminal_event": self.terminal_event.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class GroundedAssistantSegment:
    """Symbolic assistant text with an explicit terminal-result dependency."""

    segment_id: str
    text: str
    ref: int
    start_frame: int
    end_frame_exclusive: int
    terminal_event_id: str
    terminal_visible_frame: int

    def __post_init__(self) -> None:
        if not isinstance(self.segment_id, str) or not self.segment_id:
            raise CorpusV2Error("grounded segment_id is required")
        if not isinstance(self.text, str) or not self.text:
            raise CorpusV2Error("grounded segment text is required")
        _require_int("grounded segment REF", self.ref, minimum=0)
        start = _require_int("grounded segment start_frame", self.start_frame, minimum=0)
        end = _require_int("grounded segment end_frame_exclusive", self.end_frame_exclusive, minimum=1)
        visible = _require_int(
            "grounded segment terminal_visible_frame", self.terminal_visible_frame, minimum=1
        )
        if not isinstance(self.terminal_event_id, str) or not self.terminal_event_id:
            raise CorpusV2Error("grounded segment requires a terminal dependency event")
        if start < visible:
            raise CorpusV2Error(
                "grounded assistant segment cannot begin before terminal result visible_frame"
            )
        if end <= start:
            raise CorpusV2Error("grounded segment must occupy at least one frame")

    @property
    def causal_gap_frames(self) -> int:
        return self.start_frame - self.terminal_visible_frame

    def to_dict(self) -> dict[str, object]:
        return {
            "segment_id": self.segment_id,
            "text": self.text,
            "ref": self.ref,
            "start_frame": self.start_frame,
            "end_frame_exclusive": self.end_frame_exclusive,
            "dependency": {
                "terminal_event_id": self.terminal_event_id,
                "terminal_visible_frame": self.terminal_visible_frame,
                "constraint": "start_frame >= terminal_visible_frame",
                "causal_gap_frames": self.causal_gap_frames,
                "satisfied": True,
            },
        }


@dataclass(frozen=True, slots=True)
class SymbolicCorpusRecord:
    """One immutable, fingerprinted, aligned symbolic training scenario."""

    spec: ScenarioSpec
    manifest_fingerprint: str
    action_lane: MicroLane
    environment_schedule: EnvironmentSchedule
    executions: tuple[SyntheticCallTrace, ...]
    grounded_segments: tuple[GroundedAssistantSegment, ...]
    base_audio_codes: BaseAudioCodesContract
    format_version: str = CORPUS_V2_VERSION

    def __post_init__(self) -> None:
        if self.format_version != CORPUS_V2_VERSION:
            raise CorpusV2Error(f"symbolic corpus format is fixed at {CORPUS_V2_VERSION}")
        if self.manifest_fingerprint != DEFAULT_MANIFEST.fingerprint:
            raise CorpusV2Error("symbolic corpus manifest fingerprint is not the fixed V2 contract")
        executions = tuple(self.executions)
        segments = tuple(self.grounded_segments)
        if len(executions) != self.spec.call_count or len(segments) != self.spec.call_count:
            raise CorpusV2Error("symbolic corpus requires one execution and grounding per call")
        if self.action_lane.frame_count != self.spec.frame_count:
            raise CorpusV2Error("action lane does not span the scenario frame count")
        if self.environment_schedule.frame_count != self.spec.frame_count:
            raise CorpusV2Error("environment lane does not span the scenario frame count")
        if self.environment_schedule.manifest_fingerprint != self.manifest_fingerprint:
            raise CorpusV2Error("environment schedule uses a different V2 manifest")
        if self.base_audio_codes.shape != (BASE_AUDIO_CODEBOOKS, self.spec.frame_count):
            raise CorpusV2Error("base audio placeholder is not aligned [17,T]")

        expected_action = materialize_action_lane(
            self.spec.frame_count,
            (execution.packet for execution in executions),
            manifest=DEFAULT_MANIFEST,
        )
        if expected_action != self.action_lane:
            raise CorpusV2Error("action lane is not the materialization of its call packets")

        terminal_by_id = {
            scheduled.event.event_id: scheduled
            for scheduled in self.environment_schedule.scheduled_events
            if scheduled.is_terminal
        }
        if len(terminal_by_id) != self.spec.call_count:
            raise CorpusV2Error("symbolic corpus requires one scheduled terminal result per call")
        for execution, segment in zip(executions, segments, strict=True):
            if execution.call_index != len(
                [item for item in executions if item.call_index < execution.call_index]
            ):
                raise CorpusV2Error("execution call indices must be contiguous from zero")
            try:
                terminal = terminal_by_id[segment.terminal_event_id]
            except KeyError as exc:
                raise CorpusV2Error("grounded segment references an unknown terminal event") from exc
            if terminal.event.event_id != execution.terminal_event.event_id:
                raise CorpusV2Error("grounded segment dependency does not match its execution")
            if segment.ref != execution.lease.ref or segment.terminal_visible_frame != terminal.visible_frame:
                raise CorpusV2Error("grounded segment causal metadata does not match the schedule")
            if segment.end_frame_exclusive > self.spec.frame_count:
                raise CorpusV2Error("grounded segment exceeds the aligned scenario timeline")
        object.__setattr__(self, "executions", executions)
        object.__setattr__(self, "grounded_segments", segments)

    @property
    def scenario_id(self) -> str:
        return self.spec.scenario_id

    @property
    def fingerprint(self) -> str:
        return _sha256(self.to_dict())

    @property
    def action_ids(self) -> tuple[tuple[int, ...], ...]:
        return self.action_lane.ids

    @property
    def action_mask(self) -> tuple[tuple[bool, ...], ...]:
        return self.action_lane.mask

    @property
    def environment_ids(self) -> tuple[tuple[int, ...], ...]:
        return self.environment_schedule.ids

    @property
    def environment_mask(self) -> tuple[tuple[bool, ...], ...]:
        return self.environment_schedule.mask

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "scenario_id": self.scenario_id,
            "manifest_fingerprint": self.manifest_fingerprint,
            "spec": self.spec.to_dict(),
            "action_lane": self.action_lane.to_dict(),
            "environment_schedule": self.environment_schedule.to_dict(),
            "executions": [execution.to_dict() for execution in self.executions],
            "grounded_segments": [segment.to_dict() for segment in self.grounded_segments],
            "base_audio_codes": self.base_audio_codes.to_dict(),
        }

    def canonical_bytes(self) -> bytes:
        """Canonical body bytes used by fingerprints and replay comparison."""

        return _canonical_bytes(self.to_dict())

    def serialize_bytes(self) -> bytes:
        """Canonical self-identifying envelope for storage or transfer."""

        return _canonical_bytes({"fingerprint": self.fingerprint, "record": self.to_dict()})

    def model_visible_prefix_bytes(self, end_frame_exclusive: int) -> bytes:
        """Return the aligned token/text prefix available before a frame cut.

        Hidden UUID/lease metadata, scenario-generation settings, and future
        executor events are intentionally excluded.  Environment frame ``f``
        becomes usable by model output at ``f + 1``; callers comparing a future
        result counterfactual should cut at or before its ``injected_start``.
        """

        end = _require_int("prefix end_frame_exclusive", end_frame_exclusive, minimum=0)
        if end > self.spec.frame_count:
            raise CorpusV2Error("model-visible prefix exceeds the scenario timeline")
        visible_segments = [
            segment.to_dict() for segment in self.grounded_segments if segment.start_frame < end
        ]
        return _canonical_bytes(
            {
                "frame_count": end,
                "action_ids": [list(frame) for frame in self.action_ids[:end]],
                "action_mask": [list(frame) for frame in self.action_mask[:end]],
                "environment_ids": [list(frame) for frame in self.environment_ids[:end]],
                "environment_mask": [list(frame) for frame in self.environment_mask[:end]],
                "grounded_segments": visible_segments,
                "base_audio_codes": {
                    "shape": [BASE_AUDIO_CODEBOOKS, end],
                    "materialized": False,
                },
            }
        )


class SymbolicCorpusBuilder:
    """Compile a :class:`ScenarioSpec` into aligned V2 action/result lanes."""

    def __init__(
        self,
        manifest: ActionV2Manifest = DEFAULT_MANIFEST,
        executor: FiniteDomainSyntheticExecutor | None = None,
    ) -> None:
        if manifest.fingerprint != DEFAULT_MANIFEST.fingerprint:
            raise CorpusV2Error("symbolic V2 builder requires the fixed action manifest")
        self.manifest = manifest
        self.compiler = ActionCompiler(manifest)
        self.parser = ActionParser(manifest)
        self.executor = executor or FiniteDomainSyntheticExecutor(manifest)
        if self.executor.manifest.fingerprint != manifest.fingerprint:
            raise CorpusV2Error("synthetic executor and corpus builder manifests differ")

    def build(self, spec: ScenarioSpec) -> SymbolicCorpusRecord:
        if not isinstance(spec, ScenarioSpec):
            raise CorpusV2Error("symbolic corpus builder requires a ScenarioSpec")
        lease_guard = RefLeaseGuard(self.manifest)
        packets: list[ActionPacketV2] = []
        events: list[EnvironmentEventV2] = []
        executions: list[SyntheticCallTrace] = []

        for call_index in range(spec.call_count):
            tool_name = spec.tool_name(call_index)
            arguments = self._arguments(spec, call_index, tool_name)
            packet = self.compiler.compile_call(
                spec.packet_start_frame(call_index),
                call_index,
                tool_name,
                arguments,
                active_refs=lease_guard,
            )
            parsed = self.parser.parse(packet, active_refs=lease_guard)
            call_uuid = str(
                uuid.uuid5(
                    _UUID_NAMESPACE,
                    _canonical_json(
                        {
                            "scenario_id": spec.scenario_id,
                            "call_index": call_index,
                            "packet": packet.to_dict(self.manifest),
                        }
                    ),
                )
            )
            lease = lease_guard.reserve_parsed(parsed, call_uuid=call_uuid)
            outcome = self.executor.execute(parsed, spec=spec, call_index=call_index)

            # Event timing and identity are derived from the authoritative lease,
            # never copied from untrusted executor input.
            pending = EnvironmentEventV2(
                event_id=f"{spec.scenario_id}:call-{call_index}:pending",
                event_seq=call_index * 2,
                available_frame=lease.issued_frame + spec.pending_delay_frames,
                issued_frame=lease.issued_frame,
                ref=lease.ref,
                tool_name=tool_name,
                kind=EnvironmentKind.PENDING,
                call_uuid=lease.call_uuid,
                lease=lease.lease,
                manifest=self.manifest,
            )
            terminal = EnvironmentEventV2(
                event_id=f"{spec.scenario_id}:call-{call_index}:terminal",
                event_seq=call_index * 2 + 1,
                available_frame=lease.issued_frame + spec.terminal_latency(call_index),
                issued_frame=lease.issued_frame,
                ref=lease.ref,
                tool_name=tool_name,
                kind=outcome.kind,
                payload_tokens=outcome.payload_tokens,
                call_uuid=lease.call_uuid,
                lease=lease.lease,
                manifest=self.manifest,
            )
            packets.append(packet)
            events.extend((pending, terminal))
            executions.append(
                SyntheticCallTrace(
                    call_index=call_index,
                    packet=packet,
                    parsed_action=parsed,
                    lease=lease,
                    outcome=outcome,
                    pending_event=pending,
                    terminal_event=terminal,
                )
            )

        action_lane = materialize_action_lane(
            spec.frame_count,
            packets,
            manifest=self.manifest,
        )
        environment_schedule = EnvironmentFifoScheduler(self.manifest).schedule(
            spec.frame_count,
            events,
            lease_guard=lease_guard,
        )
        self._acknowledge_scheduled_visibility(environment_schedule, lease_guard)

        grounded: list[GroundedAssistantSegment] = []
        for execution in executions:
            scheduled_terminal = environment_schedule.event(execution.terminal_event.event_id)
            start = scheduled_terminal.visible_frame + spec.grounded_response_delay_frames
            if start >= spec.frame_count:
                raise CorpusV2Error(
                    "scenario timeline ends before a terminal result can ground assistant output"
                )
            grounded.append(
                GroundedAssistantSegment(
                    segment_id=f"{spec.scenario_id}:assistant-{execution.call_index}",
                    text=execution.outcome.grounded_text,
                    ref=execution.lease.ref,
                    start_frame=start,
                    end_frame_exclusive=start + 1,
                    terminal_event_id=execution.terminal_event.event_id,
                    terminal_visible_frame=scheduled_terminal.visible_frame,
                )
            )

        return SymbolicCorpusRecord(
            spec=spec,
            manifest_fingerprint=self.manifest.fingerprint,
            action_lane=action_lane,
            environment_schedule=environment_schedule,
            executions=tuple(executions),
            grounded_segments=tuple(grounded),
            base_audio_codes=BaseAudioCodesContract(spec.frame_count),
        )

    @staticmethod
    def _acknowledge_scheduled_visibility(
        schedule: EnvironmentSchedule,
        lease_guard: RefLeaseGuard,
    ) -> None:
        """Advance trusted offline time and release only visible terminals.

        Schedule construction is intentionally side-effect free.  The corpus
        builder explicitly advances its offline clock to the last terminal's
        causal visibility frame and asks the authenticated schedule to release
        the matching leases.  It never manufactures receipts in this module.
        """

        final_frame = max(
            (event.visible_frame for event in schedule.scheduled_events if event.is_terminal),
            default=0,
        )
        released = schedule.release_visible_refs(
            lease_guard,
            current_frame=final_frame,
        )
        if len(released) != len(schedule.terminal_receipts) or lease_guard.active_refs:
            raise CorpusV2Error(
                "offline visibility acknowledgement did not release every terminal REF lease"
            )

    @staticmethod
    def _arguments(spec: ScenarioSpec, call_index: int, tool_name: str) -> Mapping[str, str]:
        if tool_name == "weather.lookup":
            return MappingProxyType(
                {"city": f"CITY_{_prf(spec.seed, 'arg.weather.city', call_index) % 32}"}
            )
        if tool_name == "timer.create":
            return MappingProxyType(
                {"minutes": f"MIN_{1 + _prf(spec.seed, 'arg.timer.minutes', call_index) % 60}"}
            )
        if tool_name == "home.set_temperature":
            return MappingProxyType(
                {
                    "room": f"ROOM_{_prf(spec.seed, 'arg.home.room', call_index) % 8}",
                    "temperature_c": (
                        f"TEMP_C_{10 + _prf(spec.seed, 'arg.home.temperature', call_index) % 31}"
                    ),
                }
            )
        raise CorpusV2Error(f"tool {tool_name!r} is outside the fixed synthetic catalog")


@dataclass(frozen=True, slots=True)
class ReplayVerification:
    """Successful replay attestation."""

    scenario_id: str
    fingerprint: str
    byte_count: int
    verified: bool = True


class CorpusReplayVerifier:
    """Rebuild corpus data from its spec and reject any deterministic drift."""

    def __init__(self, builder: SymbolicCorpusBuilder | None = None) -> None:
        self.builder = builder or SymbolicCorpusBuilder()

    def verify(self, record: SymbolicCorpusRecord) -> ReplayVerification:
        if not isinstance(record, SymbolicCorpusRecord):
            raise ReplayValidationError("replay verifier requires a SymbolicCorpusRecord")
        expected = self.builder.build(record.spec)
        if record.canonical_bytes() != expected.canonical_bytes():
            differing = self._differing_sections(record.to_dict(), expected.to_dict())
            raise ReplayValidationError(
                f"deterministic corpus replay mismatch in section(s): {', '.join(differing)}"
            )
        return ReplayVerification(
            scenario_id=record.scenario_id,
            fingerprint=record.fingerprint,
            byte_count=len(record.serialize_bytes()),
        )

    def verify_bytes(self, payload: bytes | bytearray | memoryview) -> ReplayVerification:
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise ReplayValidationError("serialized corpus payload must be bytes-like")
        raw = bytes(payload)
        try:
            envelope = json.loads(raw.decode("utf-8"), object_pairs_hook=self._unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError, ReplayValidationError) as exc:
            if isinstance(exc, ReplayValidationError):
                raise
            raise ReplayValidationError("serialized corpus is not canonical-compatible JSON") from exc
        if not isinstance(envelope, dict) or set(envelope) != {"fingerprint", "record"}:
            raise ReplayValidationError("serialized corpus envelope must contain fingerprint and record")
        fingerprint = envelope["fingerprint"]
        body = envelope["record"]
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ReplayValidationError("serialized corpus fingerprint is invalid")
        if not isinstance(body, dict):
            raise ReplayValidationError("serialized corpus record must be an object")
        if _sha256(body) != fingerprint:
            raise ReplayValidationError("serialized corpus fingerprint does not match its record")
        try:
            spec = ScenarioSpec.from_dict(body["spec"])
        except (KeyError, CorpusV2Error) as exc:
            raise ReplayValidationError("serialized corpus contains an invalid scenario spec") from exc
        expected = self.builder.build(spec)
        if body != expected.to_dict():
            differing = self._differing_sections(body, expected.to_dict())
            raise ReplayValidationError(
                "serialized corpus was rehashed but fails deterministic replay in section(s): "
                + ", ".join(differing)
            )
        return ReplayVerification(
            scenario_id=expected.scenario_id,
            fingerprint=fingerprint,
            byte_count=len(raw),
        )

    @staticmethod
    def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ReplayValidationError(f"serialized corpus has duplicate JSON key {key!r}")
            result[key] = value
        return result

    @staticmethod
    def _differing_sections(
        actual: Mapping[str, object], expected: Mapping[str, object]
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                key
                for key in set(actual) | set(expected)
                if actual.get(key) != expected.get(key)
            )
        ) or ("unknown",)


__all__ = [
    "BASE_AUDIO_CODEBOOKS",
    "CORPUS_V2_VERSION",
    "BaseAudioCodesContract",
    "CorpusReplayVerifier",
    "CorpusV2Error",
    "FiniteDomainSyntheticExecutor",
    "GroundedAssistantSegment",
    "ReplayValidationError",
    "ReplayVerification",
    "ScenarioSpec",
    "SymbolicCorpusBuilder",
    "SymbolicCorpusRecord",
    "SyntheticCallTrace",
    "SyntheticToolOutcome",
]
