"""Fixed V2 symbolic action contract for PersonaPlex-Agent.

This module deliberately does *not* reuse the compatibility ``ActionPacket``
or ``TypedLaneVocabulary`` classes.  V2 has a bounded micro-decoder grammar,
an immutable 256-token vocabulary, and model-visible reference slots instead
of model-generated JSON/call UUIDs.

The model produces action tokens only.  A runtime may reserve a ``REF_n``
slot after a parsed ``CALL_END`` and associate it with a hidden UUID/lease, but
that metadata is never encoded into the model-action lane.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Any, Iterable, Mapping


ACTION_V2_VERSION = "ppx-action-v2.0.0"
BASE_FRAME_MS = 80
MICRO_TOKENS_PER_FRAME = 5
PACKET_FRAMES = 2
MAX_ACTION_TOKENS_PER_PACKET = MICRO_TOKENS_PER_FRAME * PACKET_FRAMES
MAX_REFS = 4
PAD_TOKEN_ID = -1

# Aliases make the V2 clock explicit while keeping it easy to compare against
# the existing native lane constants.
ACTION_PACKET_FRAMES = PACKET_FRAMES
MAX_ACTION_TOKENS_PER_FRAME = MICRO_TOKENS_PER_FRAME


class ActionProtocolError(ValueError):
    """Base error for a malformed V2 model-action packet."""


class ActionGrammarError(ActionProtocolError):
    """Raised when a packet is outside the bounded action FSA."""


class RefLeaseError(ActionProtocolError):
    """Raised when a model-visible REF slot is busy or has a stale lease."""


class ActionKind(str, Enum):
    """The complete set of action packet productions."""

    NOOP = "noop"
    CALL = "call"
    CANCEL = "cancel"
    CONTROL = "control"


class ControlAction(str, Enum):
    """Single-token non-tool actions permitted by the grammar."""

    BACKCHANNEL = "BACKCHANNEL"
    PAUSE = "PAUSE"
    INTERRUPT = "INTERRUPT"
    REQUEST_CONFIRM = "REQUEST_CONFIRM"


def _canonical_json(value: object) -> str:
    """Return the one canonical representation used for all V2 hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _fixed_action_tokens() -> dict[str, int]:
    tokens: dict[str, int] = {
        "NOOP": 0,
        "CALL_BEGIN": 1,
        "CALL_END": 2,
        "CANCEL": 3,
        "ACTION_END": 4,
        "BACKCHANNEL": 5,
        "PAUSE": 6,
        "INTERRUPT": 7,
        "REQUEST_CONFIRM": 8,
    }
    for ref in range(MAX_REFS):
        tokens[f"REF_{ref}"] = 9 + ref
    tokens.update(
        {
            "TOOL_WEATHER_LOOKUP": 13,
            "TOOL_TIMER_CREATE": 14,
            "TOOL_HOME_SET_TEMPERATURE": 15,
        }
    )
    for city in range(32):
        tokens[f"CITY_{city}"] = 16 + city
    for minutes in range(1, 61):
        tokens[f"MIN_{minutes}"] = 48 + (minutes - 1)
    for room in range(8):
        tokens[f"ROOM_{room}"] = 108 + room
    for temperature in range(10, 41):
        tokens[f"TEMP_C_{temperature}"] = 116 + (temperature - 10)
    for enum_value in range(45):
        tokens[f"ENUM_{enum_value}"] = 147 + enum_value
    for token_id in range(192, 256):
        tokens[f"RESERVED_{token_id}"] = token_id
    _validate_fixed_mapping(tokens, lane="action")
    return tokens


def _fixed_environment_tokens() -> dict[str, int]:
    tokens: dict[str, int] = {
        "OBS_BEGIN": 0,
        "OBS_END": 1,
        "PENDING": 2,
        "OK": 3,
        "ERROR": 4,
        "TIMEOUT": 5,
        "CANCELLED": 6,
        "STALE": 7,
        "OBS_MORE": 8,
        "ERR_REF_BUSY": 9,
        "ERR_SCHEMA": 10,
        "ERR_PERMISSION": 11,
        "ERR_EXECUTOR": 12,
    }
    for ref in range(MAX_REFS):
        tokens[f"REF_{ref}"] = 13 + ref
    tokens.update(
        {
            "TOOL_WEATHER": 17,
            "TOOL_TIMER": 18,
            "TOOL_HOME": 19,
        }
    )
    for forecast in range(16):
        tokens[f"FORECAST_{forecast:02d}"] = 20 + forecast
    for temperature in range(10, 41):
        tokens[f"TEMP_C_{temperature}"] = 36 + (temperature - 10)
    for minutes in range(1, 61):
        tokens[f"MIN_{minutes}"] = 67 + (minutes - 1)
    for room in range(8):
        tokens[f"ROOM_{room}"] = 127 + room
    for enum_value in range(57):
        tokens[f"ENUM_{enum_value}"] = 135 + enum_value
    for token_id in range(192, 256):
        tokens[f"RESERVED_{token_id}"] = token_id
    _validate_fixed_mapping(tokens, lane="environment")
    return tokens


def _validate_fixed_mapping(mapping: Mapping[str, int], *, lane: str) -> None:
    """Validate a complete, bijective V2 256-entry vocabulary map."""

    if len(mapping) != 256:
        raise ValueError(f"{lane} V2 vocabulary must contain exactly 256 tokens")
    if any(not isinstance(name, str) or not name for name in mapping):
        raise ValueError(f"{lane} V2 vocabulary contains an invalid token name")
    values = list(mapping.values())
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError(f"{lane} V2 vocabulary IDs must be integers")
    if sorted(values) != list(range(256)):
        raise ValueError(f"{lane} V2 vocabulary IDs must be the exact range 0..255")


@dataclass(frozen=True, slots=True)
class ToolArgumentSpec:
    """A fixed positional symbolic argument accepted by one tool."""

    name: str
    allowed_tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool argument name is required")
        tokens = tuple(self.allowed_tokens)
        if not tokens or any(not isinstance(token, str) or not token for token in tokens):
            raise ValueError(f"tool argument {self.name!r} requires symbolic allowed tokens")
        if len(tokens) != len(set(tokens)):
            raise ValueError(f"tool argument {self.name!r} has duplicate allowed tokens")
        object.__setattr__(self, "allowed_tokens", tokens)

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "allowed_tokens": list(self.allowed_tokens)}


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool's model action and environment observation vocabulary binding."""

    name: str
    action_token: str
    environment_token: str
    arguments: tuple[ToolArgumentSpec, ...]
    result_tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.action_token or not self.environment_token:
            raise ValueError("tool name and lane tokens are required")
        arguments = tuple(self.arguments)
        if not all(isinstance(argument, ToolArgumentSpec) for argument in arguments):
            raise ValueError("tool arguments must be ToolArgumentSpec values")
        if len({argument.name for argument in arguments}) != len(arguments):
            raise ValueError(f"tool {self.name!r} has duplicate argument names")
        result_tokens = tuple(self.result_tokens)
        if not result_tokens or any(not isinstance(token, str) or not token for token in result_tokens):
            raise ValueError(f"tool {self.name!r} requires symbolic result tokens")
        if len(result_tokens) != len(set(result_tokens)):
            raise ValueError(f"tool {self.name!r} has duplicate result tokens")
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(self, "result_tokens", result_tokens)

    def argument(self, name: str) -> ToolArgumentSpec:
        for argument in self.arguments:
            if argument.name == name:
                return argument
        raise ValueError(f"tool {self.name!r} has no argument {name!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "action_token": self.action_token,
            "environment_token": self.environment_token,
            "arguments": [argument.to_dict() for argument in self.arguments],
            "result_tokens": list(self.result_tokens),
        }


def _default_tool_specs() -> tuple[ToolSpec, ...]:
    return (
        ToolSpec(
            name="weather.lookup",
            action_token="TOOL_WEATHER_LOOKUP",
            environment_token="TOOL_WEATHER",
            arguments=(
                ToolArgumentSpec("city", tuple(f"CITY_{city}" for city in range(32))),
            ),
            result_tokens=tuple(f"FORECAST_{forecast:02d}" for forecast in range(16))
            + tuple(f"TEMP_C_{temperature}" for temperature in range(10, 41)),
        ),
        ToolSpec(
            name="timer.create",
            action_token="TOOL_TIMER_CREATE",
            environment_token="TOOL_TIMER",
            arguments=(
                ToolArgumentSpec("minutes", tuple(f"MIN_{minute}" for minute in range(1, 61))),
            ),
            result_tokens=("ENUM_0",),
        ),
        ToolSpec(
            name="home.set_temperature",
            action_token="TOOL_HOME_SET_TEMPERATURE",
            environment_token="TOOL_HOME",
            arguments=(
                ToolArgumentSpec("room", tuple(f"ROOM_{room}" for room in range(8))),
                ToolArgumentSpec(
                    "temperature_c", tuple(f"TEMP_C_{temperature}" for temperature in range(10, 41))
                ),
            ),
            result_tokens=("ENUM_1",),
        ),
    )


_DEFAULT_ACTION_TOKENS = _fixed_action_tokens()
_DEFAULT_ENVIRONMENT_TOKENS = _fixed_environment_tokens()
_DEFAULT_TOOL_SPECS = _default_tool_specs()


@dataclass(frozen=True, slots=True)
class ActionV2Manifest:
    """An immutable, fingerprinted specification for the V2 typed lanes.

    The maps are copied into :class:`types.MappingProxyType` objects so an
    external caller cannot mutate the vocabulary after a checkpoint or corpus
    manifest records its fingerprint.
    """

    version: str = ACTION_V2_VERSION
    base_frame_ms: int = BASE_FRAME_MS
    micro_tokens_per_frame: int = MICRO_TOKENS_PER_FRAME
    packet_frames: int = PACKET_FRAMES
    max_packet_tokens: int = MAX_ACTION_TOKENS_PER_PACKET
    max_refs: int = MAX_REFS
    action_tokens: Mapping[str, int] = field(default_factory=_fixed_action_tokens)
    environment_tokens: Mapping[str, int] = field(default_factory=_fixed_environment_tokens)
    tool_schemas: tuple[ToolSpec, ...] = field(default_factory=_default_tool_specs)
    _action_names: Mapping[int, str] = field(init=False, repr=False, compare=False)
    _environment_names: Mapping[int, str] = field(init=False, repr=False, compare=False)
    _tools_by_name: Mapping[str, ToolSpec] = field(init=False, repr=False, compare=False)
    _tools_by_action_token: Mapping[str, ToolSpec] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.version != ACTION_V2_VERSION:
            raise ValueError(f"this fixed action contract requires version {ACTION_V2_VERSION!r}")
        if self.base_frame_ms != BASE_FRAME_MS:
            raise ValueError(f"V2 action clock is fixed at {BASE_FRAME_MS} ms")
        if self.micro_tokens_per_frame != MICRO_TOKENS_PER_FRAME:
            raise ValueError(
                f"V2 action lane is fixed at {MICRO_TOKENS_PER_FRAME} micro-slots/frame"
            )
        if self.packet_frames != PACKET_FRAMES:
            raise ValueError(f"V2 action packet is fixed at {PACKET_FRAMES} frames")
        if self.max_packet_tokens != MAX_ACTION_TOKENS_PER_PACKET:
            raise ValueError(
                f"V2 action packet is fixed at {MAX_ACTION_TOKENS_PER_PACKET} tokens"
            )
        if self.max_refs != MAX_REFS:
            raise ValueError(f"V2 action protocol is fixed at {MAX_REFS} reference slots")

        action_tokens = dict(self.action_tokens)
        environment_tokens = dict(self.environment_tokens)
        _validate_fixed_mapping(action_tokens, lane="action")
        _validate_fixed_mapping(environment_tokens, lane="environment")
        if action_tokens != _DEFAULT_ACTION_TOKENS:
            raise ValueError("V2 action vocabulary is fixed; use a new protocol version for changes")
        if environment_tokens != _DEFAULT_ENVIRONMENT_TOKENS:
            raise ValueError("V2 environment vocabulary is fixed; use a new protocol version for changes")

        tools = tuple(self.tool_schemas)
        if not tools:
            raise ValueError("V2 manifest requires at least one tool schema")
        if not all(isinstance(tool, ToolSpec) for tool in tools):
            raise ValueError("tool_schemas must contain ToolSpec values")
        tools_by_name = {tool.name: tool for tool in tools}
        tools_by_action = {tool.action_token: tool for tool in tools}
        if len(tools_by_name) != len(tools) or len(tools_by_action) != len(tools):
            raise ValueError("V2 tool schemas must have unique names and action tokens")
        if tools != _DEFAULT_TOOL_SPECS:
            raise ValueError("V2 tool catalog is fixed; use a new protocol version for changes")
        for tool in tools:
            if tool.action_token not in action_tokens:
                raise ValueError(f"tool {tool.name!r} uses an unknown action token")
            if tool.environment_token not in environment_tokens:
                raise ValueError(f"tool {tool.name!r} uses an unknown environment token")
            for argument in tool.arguments:
                if any(token not in action_tokens for token in argument.allowed_tokens):
                    raise ValueError(f"tool {tool.name!r} has an unknown action argument token")
            if any(token not in environment_tokens for token in tool.result_tokens):
                raise ValueError(f"tool {tool.name!r} has an unknown environment result token")
            # CALL_BEGIN + TOOL + REF + args + CALL_END must fit one packet.
            if 4 + len(tool.arguments) > self.max_packet_tokens:
                raise ValueError(f"tool {tool.name!r} exceeds the V2 action packet budget")

        object.__setattr__(self, "action_tokens", MappingProxyType(action_tokens))
        object.__setattr__(self, "environment_tokens", MappingProxyType(environment_tokens))
        object.__setattr__(self, "tool_schemas", tools)
        object.__setattr__(
            self,
            "_action_names",
            MappingProxyType({token_id: token for token, token_id in action_tokens.items()}),
        )
        object.__setattr__(
            self,
            "_environment_names",
            MappingProxyType({token_id: token for token, token_id in environment_tokens.items()}),
        )
        object.__setattr__(self, "_tools_by_name", MappingProxyType(tools_by_name))
        object.__setattr__(self, "_tools_by_action_token", MappingProxyType(tools_by_action))

    @property
    def action_cardinality(self) -> int:
        return len(self.action_tokens)

    @property
    def environment_cardinality(self) -> int:
        return len(self.environment_tokens)

    @property
    def action_names(self) -> Mapping[int, str]:
        return self._action_names

    @property
    def environment_names(self) -> Mapping[int, str]:
        return self._environment_names

    @property
    def tool_catalog_hash(self) -> str:
        return _sha256({"tools": [tool.to_dict() for tool in self.tool_schemas]})

    @property
    def fingerprint(self) -> str:
        return _sha256(self.to_dict())

    @property
    def vocabulary_fingerprint(self) -> str:
        """Alias retained for checkpoint/corpus manifest terminology."""

        return self.fingerprint

    def action_id(self, token: str) -> int:
        try:
            return self.action_tokens[token]
        except KeyError as exc:
            raise ValueError(f"unknown V2 action token: {token!r}") from exc

    def environment_id(self, token: str) -> int:
        try:
            return self.environment_tokens[token]
        except KeyError as exc:
            raise ValueError(f"unknown V2 environment token: {token!r}") from exc

    def action_name(self, token_id: int) -> str:
        try:
            return self._action_names[token_id]
        except KeyError as exc:
            raise ActionGrammarError(f"unknown V2 action token ID: {token_id!r}") from exc

    def environment_name(self, token_id: int) -> str:
        try:
            return self._environment_names[token_id]
        except KeyError as exc:
            raise ValueError(f"unknown V2 environment token ID: {token_id!r}") from exc

    def tool(self, name: str) -> ToolSpec:
        try:
            return self._tools_by_name[name]
        except KeyError as exc:
            raise ValueError(f"unknown V2 tool: {name!r}") from exc

    def tool_for_action_token(self, token: str) -> ToolSpec:
        try:
            return self._tools_by_action_token[token]
        except KeyError as exc:
            raise ActionGrammarError(f"action token {token!r} is not a callable V2 tool") from exc

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-safe snapshot; callers cannot mutate this manifest."""

        return {
            "version": self.version,
            "base_frame_ms": self.base_frame_ms,
            "micro_tokens_per_frame": self.micro_tokens_per_frame,
            "packet_frames": self.packet_frames,
            "max_packet_tokens": self.max_packet_tokens,
            "max_refs": self.max_refs,
            "action_tokens": dict(self.action_tokens),
            "environment_tokens": dict(self.environment_tokens),
            "tool_schemas": [tool.to_dict() for tool in self.tool_schemas],
            "tool_catalog_hash": self.tool_catalog_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionV2Manifest":
        """Reconstruct a manifest snapshot while revalidating every invariant."""

        try:
            tools = tuple(
                ToolSpec(
                    name=item["name"],
                    action_token=item["action_token"],
                    environment_token=item["environment_token"],
                    arguments=tuple(
                        ToolArgumentSpec(
                            name=argument["name"],
                            allowed_tokens=tuple(argument["allowed_tokens"]),
                        )
                        for argument in item["arguments"]
                    ),
                    result_tokens=tuple(item["result_tokens"]),
                )
                for item in data["tool_schemas"]
            )
            manifest = cls(
                version=data["version"],
                base_frame_ms=data["base_frame_ms"],
                micro_tokens_per_frame=data["micro_tokens_per_frame"],
                packet_frames=data["packet_frames"],
                max_packet_tokens=data["max_packet_tokens"],
                max_refs=data["max_refs"],
                action_tokens=data["action_tokens"],
                environment_tokens=data["environment_tokens"],
                tool_schemas=tools,
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid V2 action manifest snapshot") from exc
        expected_hash = data.get("tool_catalog_hash")
        if expected_hash is not None and expected_hash != manifest.tool_catalog_hash:
            raise ValueError("V2 manifest tool_catalog_hash does not match its schemas")
        return manifest


DEFAULT_MANIFEST = ActionV2Manifest()


def default_manifest() -> ActionV2Manifest:
    """Return the immutable fixed V2 manifest singleton."""

    return DEFAULT_MANIFEST


def _normalise_ref(ref: int | str, *, max_refs: int = MAX_REFS) -> int:
    if isinstance(ref, bool):
        raise ValueError("REF must be an integer index or REF_n token")
    if isinstance(ref, int):
        index = ref
    elif isinstance(ref, str) and ref.startswith("REF_"):
        suffix = ref.removeprefix("REF_")
        try:
            index = int(suffix)
        except ValueError as exc:
            raise ValueError(f"invalid REF token: {ref!r}") from exc
    else:
        raise ValueError("REF must be an integer index or REF_n token")
    if index < 0 or index >= max_refs:
        raise ValueError(f"REF_{index} is outside the 0..{max_refs - 1} V2 range")
    return index


def _active_ref_set(active_refs: object | None, *, max_refs: int) -> frozenset[int] | None:
    """Normalize a runtime guard, mapping, or iterable to model-visible refs."""

    if active_refs is None:
        return None
    candidate = getattr(active_refs, "active_refs", active_refs)
    if isinstance(candidate, Mapping):
        candidate = candidate.keys()
    try:
        return frozenset(_normalise_ref(ref, max_refs=max_refs) for ref in candidate)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("active_refs must be an iterable, mapping, or RefLeaseGuard") from exc


@dataclass(frozen=True, slots=True)
class ActionPacketV2:
    """One bounded two-frame logical model-action packet.

    ``token_ids`` contains only semantic action IDs.  ``PAD_TOKEN_ID`` is a
    materialized-lane filler and is intentionally forbidden here so the parser
    never mistakes masked cache state for a model decision.
    """

    packet_start_frame: int
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.packet_start_frame, bool) or not isinstance(self.packet_start_frame, int):
            raise ValueError("packet_start_frame must be an integer")
        if self.packet_start_frame < 0 or self.packet_start_frame % PACKET_FRAMES:
            raise ValueError("V2 action packets must start on an even, non-negative 80 ms frame")
        token_ids = tuple(self.token_ids)
        if not token_ids:
            raise ValueError("V2 action packets must contain at least one semantic token")
        if len(token_ids) > MAX_ACTION_TOKENS_PER_PACKET:
            raise ValueError("V2 action packet exceeds the 10-token / 160-ms budget")
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise ValueError("V2 action token IDs must be integers")
            if token_id == PAD_TOKEN_ID:
                raise ValueError("PAD=-1 is a lane filler and cannot appear in an action packet")
            if token_id < 0 or token_id >= 256:
                raise ValueError("V2 action packet token IDs must be in the fixed 0..255 range")
        object.__setattr__(self, "token_ids", token_ids)

    @property
    def start_frame(self) -> int:
        """Compatibility spelling for code that describes logical packets."""

        return self.packet_start_frame

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    def tokens_for_frame(self, frame: int) -> tuple[int, ...]:
        if frame not in (self.packet_start_frame, self.packet_start_frame + 1):
            raise ValueError("frame is outside this V2 action packet")
        start = 0 if frame == self.packet_start_frame else MICRO_TOKENS_PER_FRAME
        return self.token_ids[start : start + MICRO_TOKENS_PER_FRAME]

    def to_dict(self, manifest: ActionV2Manifest = DEFAULT_MANIFEST) -> dict[str, object]:
        return {
            "packet_start_frame": self.packet_start_frame,
            "token_ids": list(self.token_ids),
            "tokens": [manifest.action_name(token_id) for token_id in self.token_ids],
        }


@dataclass(frozen=True, slots=True)
class MicroPosition:
    """A causal location inside a 5-slot 80 ms frame."""

    frame: int
    micro: int

    def __post_init__(self) -> None:
        if self.frame < 0:
            raise ValueError("micro position frame must be non-negative")
        if self.micro < 0 or self.micro >= MICRO_TOKENS_PER_FRAME:
            raise ValueError("micro position is outside the five-slot V2 frame")


@dataclass(frozen=True, slots=True)
class ParsedAction:
    """A complete FSA parse, with no executor-facing hidden identity fields."""

    kind: ActionKind
    packet: ActionPacketV2
    issued_frame: int
    issued_micro: int
    ref: int | None = None
    tool_name: str | None = None
    arguments: Mapping[str, str] = field(default_factory=dict)
    control: str | None = None

    def __post_init__(self) -> None:
        if self.issued_frame < self.packet.packet_start_frame:
            raise ValueError("parsed action cannot issue before its packet")
        if self.issued_micro < 0 or self.issued_micro >= MICRO_TOKENS_PER_FRAME:
            raise ValueError("parsed action issued_micro is outside the V2 frame")
        arguments = dict(self.arguments)
        if any(not isinstance(name, str) or not isinstance(value, str) for name, value in arguments.items()):
            raise ValueError("parsed action arguments must be symbolic string pairs")
        object.__setattr__(self, "arguments", MappingProxyType(arguments))

    @property
    def issued_position(self) -> MicroPosition:
        return MicroPosition(self.issued_frame, self.issued_micro)

    @property
    def dispatchable(self) -> bool:
        """Only a completed grammar-valid CALL_END is eligible for dispatch."""

        return self.kind is ActionKind.CALL

    @property
    def reference(self) -> int | None:
        return self.ref

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "packet_start_frame": self.packet.packet_start_frame,
            "token_ids": list(self.packet.token_ids),
            "issued_frame": self.issued_frame,
            "issued_micro": self.issued_micro,
            "ref": self.ref,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "control": self.control,
            "dispatchable": self.dispatchable,
        }


class ActionParser:
    """Parse exactly one V2 packet against the fixed finite-state grammar."""

    def __init__(self, manifest: ActionV2Manifest = DEFAULT_MANIFEST) -> None:
        self.manifest = manifest

    def parse(
        self,
        packet: ActionPacketV2,
        *,
        active_refs: object | None = None,
    ) -> ParsedAction:
        if packet.packet_start_frame % PACKET_FRAMES:
            raise ActionGrammarError("CALL_BEGIN is legal only at an even packet frame")
        names = self._token_names(packet)
        if any(name.startswith("RESERVED_") for name in names):
            raise ActionGrammarError("reserved V2 action token IDs are grammar-forbidden")
        active = _active_ref_set(active_refs, max_refs=self.manifest.max_refs)

        if names == ("NOOP",):
            return self._parsed(ActionKind.NOOP, packet, token_index=0)

        if names[0] == "CALL_BEGIN":
            return self._parse_call(packet, names, active)

        if names[0] == "CANCEL":
            return self._parse_cancel(packet, names, active)

        if len(names) == 1 and names[0] in {control.value for control in ControlAction}:
            return self._parsed(
                ActionKind.CONTROL,
                packet,
                token_index=0,
                control=names[0],
            )

        raise ActionGrammarError(
            "V2 action packet must match NOOP, CALL_BEGIN ... CALL_END, "
            "CANCEL REF ACTION_END, or a single control token"
        )

    def _token_names(self, packet: ActionPacketV2) -> tuple[str, ...]:
        names: list[str] = []
        for token_id in packet.token_ids:
            if token_id == PAD_TOKEN_ID:
                raise ActionGrammarError("PAD=-1 cannot be parsed as a V2 action")
            names.append(self.manifest.action_name(token_id))
        return tuple(names)

    def _parse_call(
        self,
        packet: ActionPacketV2,
        names: tuple[str, ...],
        active: frozenset[int] | None,
    ) -> ParsedAction:
        if len(names) < 4 or names[-1] != "CALL_END":
            raise ActionGrammarError("a V2 call must terminate with CALL_END")
        tool = self.manifest.tool_for_action_token(names[1])
        ref = self._ref_from_action_name(names[2])
        if active is not None and ref in active:
            raise RefLeaseError(f"REF_{ref} is already active and cannot be reused")
        expected_length = 4 + len(tool.arguments)
        if len(names) != expected_length:
            raise ActionGrammarError(
                f"tool {tool.name!r} requires exactly {len(tool.arguments)} positional arguments "
                "before CALL_END"
            )
        arguments: dict[str, str] = {}
        for argument, token in zip(tool.arguments, names[3:-1], strict=True):
            if token not in argument.allowed_tokens:
                raise ActionGrammarError(
                    f"token {token!r} is not valid for positional argument {argument.name!r} "
                    f"of {tool.name!r}"
                )
            arguments[argument.name] = token
        return self._parsed(
            ActionKind.CALL,
            packet,
            token_index=len(names) - 1,
            ref=ref,
            tool_name=tool.name,
            arguments=arguments,
        )

    def _parse_cancel(
        self,
        packet: ActionPacketV2,
        names: tuple[str, ...],
        active: frozenset[int] | None,
    ) -> ParsedAction:
        if len(names) != 3 or names[-1] != "ACTION_END":
            raise ActionGrammarError("a V2 cancellation must be exactly CANCEL REF ACTION_END")
        ref = self._ref_from_action_name(names[1])
        if active is not None and ref not in active:
            raise RefLeaseError(f"REF_{ref} is not active and cannot be cancelled")
        return self._parsed(ActionKind.CANCEL, packet, token_index=2, ref=ref)

    def _ref_from_action_name(self, token: str) -> int:
        if not token.startswith("REF_"):
            raise ActionGrammarError("V2 call/cancel requires a REF_0..REF_3 token")
        try:
            return _normalise_ref(token, max_refs=self.manifest.max_refs)
        except ValueError as exc:
            raise ActionGrammarError(str(exc)) from exc

    @staticmethod
    def _parsed(
        kind: ActionKind,
        packet: ActionPacketV2,
        *,
        token_index: int,
        ref: int | None = None,
        tool_name: str | None = None,
        arguments: Mapping[str, str] | None = None,
        control: str | None = None,
    ) -> ParsedAction:
        return ParsedAction(
            kind=kind,
            packet=packet,
            issued_frame=packet.packet_start_frame + (token_index // MICRO_TOKENS_PER_FRAME),
            issued_micro=token_index % MICRO_TOKENS_PER_FRAME,
            ref=ref,
            tool_name=tool_name,
            arguments={} if arguments is None else arguments,
            control=control,
        )


class ActionCompiler:
    """Compile schema-constrained symbolic calls into one V2 action packet."""

    def __init__(self, manifest: ActionV2Manifest = DEFAULT_MANIFEST) -> None:
        self.manifest = manifest
        self.parser = ActionParser(manifest)

    def compile_call(
        self,
        packet_start_frame: int,
        ref: int | str,
        tool_name: str,
        arguments: Mapping[str, str],
        *,
        active_refs: object | None = None,
    ) -> ActionPacketV2:
        tool = self.manifest.tool(tool_name)
        ref_index = _normalise_ref(ref, max_refs=self.manifest.max_refs)
        if not isinstance(arguments, Mapping):
            raise ValueError("V2 tool arguments must be a mapping of symbolic token names")
        expected = tuple(argument.name for argument in tool.arguments)
        received = tuple(arguments.keys())
        if set(received) != set(expected) or len(received) != len(expected):
            missing = [name for name in expected if name not in arguments]
            extra = [name for name in arguments if name not in expected]
            raise ValueError(
                f"tool {tool.name!r} arguments must be exactly {expected}; "
                f"missing={missing}, extra={extra}"
            )
        argument_tokens: list[str] = []
        for argument in tool.arguments:
            value = arguments[argument.name]
            if not isinstance(value, str) or value not in argument.allowed_tokens:
                raise ValueError(
                    f"argument {argument.name!r} for {tool.name!r} must be one of its "
                    "fixed symbolic token values"
                )
            argument_tokens.append(value)
        packet = self._packet(
            packet_start_frame,
            ("CALL_BEGIN", tool.action_token, f"REF_{ref_index}", *argument_tokens, "CALL_END"),
        )
        # Compilation shares the parser's active-lease gate; no generated call
        # can accidentally be considered valid by one path but not the other.
        self.parser.parse(packet, active_refs=active_refs)
        return packet

    def compile_cancel(
        self,
        packet_start_frame: int,
        ref: int | str,
        *,
        active_refs: object | None = None,
    ) -> ActionPacketV2:
        ref_index = _normalise_ref(ref, max_refs=self.manifest.max_refs)
        packet = self._packet(packet_start_frame, ("CANCEL", f"REF_{ref_index}", "ACTION_END"))
        self.parser.parse(packet, active_refs=active_refs)
        return packet

    def compile_control(
        self,
        packet_start_frame: int,
        control: ControlAction | str,
    ) -> ActionPacketV2:
        if isinstance(control, ControlAction):
            token = control.value
        elif isinstance(control, str):
            candidate = control.upper()
            try:
                token = ControlAction(candidate).value
            except ValueError as exc:
                raise ValueError(f"unknown V2 control action: {control!r}") from exc
        else:
            raise ValueError("V2 control must be a ControlAction or token name")
        return self._packet(packet_start_frame, (token,))

    def compile_noop(self, packet_start_frame: int) -> ActionPacketV2:
        return self._packet(packet_start_frame, ("NOOP",))

    def _packet(self, packet_start_frame: int, tokens: Iterable[str]) -> ActionPacketV2:
        return ActionPacketV2(
            packet_start_frame=packet_start_frame,
            token_ids=tuple(self.manifest.action_id(token) for token in tokens),
        )


@dataclass(frozen=True, slots=True)
class RefLease:
    """Runtime-only identity attached to one model-visible reference slot."""

    ref: int
    call_uuid: str
    lease: int
    issued_frame: int
    issued_micro: int

    def __post_init__(self) -> None:
        _normalise_ref(self.ref)
        if not isinstance(self.call_uuid, str) or not self.call_uuid:
            raise ValueError("hidden call_uuid is required for a V2 REF lease")
        if isinstance(self.lease, bool) or not isinstance(self.lease, int) or self.lease < 1:
            raise ValueError("V2 REF lease counters start at one")
        if isinstance(self.issued_frame, bool) or not isinstance(self.issued_frame, int) or self.issued_frame < 0:
            raise ValueError("V2 REF lease issued_frame must be non-negative")
        if (
            isinstance(self.issued_micro, bool)
            or not isinstance(self.issued_micro, int)
            or self.issued_micro < 0
            or self.issued_micro >= MICRO_TOKENS_PER_FRAME
        ):
            raise ValueError("V2 REF lease issued_micro is outside the action frame")


@dataclass(frozen=True, slots=True)
class TerminalLeaseReceipt:
    """Trusted scheduler evidence that a terminal ``OBS_END`` was injected.

    This remains runtime metadata, not a model token packet.  The environment
    scheduler creates it only after it has placed the final ``OBS_END``
    micro-token; the guard derives model visibility from that position instead
    of accepting an arbitrary caller-supplied frame number.
    """

    ref: int
    call_uuid: str
    lease: int
    available_frame: int
    injected_end: MicroPosition
    source: str = "tool_executor"
    terminal_token: str = "OBS_END"

    def __post_init__(self) -> None:
        _normalise_ref(self.ref)
        if not isinstance(self.call_uuid, str) or not self.call_uuid:
            raise ValueError("terminal receipt requires the hidden call_uuid")
        if isinstance(self.lease, bool) or not isinstance(self.lease, int) or self.lease < 1:
            raise ValueError("terminal receipt lease must be a positive integer")
        if (
            isinstance(self.available_frame, bool)
            or not isinstance(self.available_frame, int)
            or self.available_frame < 0
        ):
            raise ValueError("terminal receipt available_frame must be non-negative")
        if not isinstance(self.injected_end, MicroPosition):
            raise ValueError("terminal receipt requires the scheduler injected_end micro-position")
        if self.injected_end.frame < self.available_frame:
            raise ValueError("terminal OBS_END cannot be injected before it is available")
        if self.source != "tool_executor":
            raise ValueError("only tool_executor observations can release a V2 REF")
        if self.terminal_token != "OBS_END":
            raise ValueError("a V2 REF can be released only by a final OBS_END receipt")

    @property
    def visible_frame(self) -> int:
        """First raw output frame causally able to use the terminal event."""

        return self.injected_end.frame + 1


class RefLeaseGuard:
    """Prevent REF reuse until a matching terminal observation becomes visible.

    The guard intentionally owns UUID/lease metadata outside the model token
    stream.  ``release_terminal`` accepts a scheduler-authenticated terminal
    ``OBS_END`` receipt, so a caller cannot recycle a slot by merely claiming a
    convenient visible frame.
    """

    def __init__(self, manifest: ActionV2Manifest = DEFAULT_MANIFEST) -> None:
        self.manifest = manifest
        self._active: dict[int, RefLease] = {}
        self._next_lease: list[int] = [0] * manifest.max_refs

    @property
    def active_refs(self) -> frozenset[int]:
        return frozenset(self._active)

    def is_active(self, ref: int | str) -> bool:
        return _normalise_ref(ref, max_refs=self.manifest.max_refs) in self._active

    def lease_for(self, ref: int | str) -> RefLease:
        ref_index = _normalise_ref(ref, max_refs=self.manifest.max_refs)
        try:
            return self._active[ref_index]
        except KeyError as exc:
            raise RefLeaseError(f"REF_{ref_index} is not active") from exc

    def reserve(
        self,
        ref: int | str,
        *,
        call_uuid: str,
        issued_frame: int,
        issued_micro: int = 0,
    ) -> RefLease:
        ref_index = _normalise_ref(ref, max_refs=self.manifest.max_refs)
        if ref_index in self._active:
            raise RefLeaseError(f"REF_{ref_index} is already active")
        self._next_lease[ref_index] += 1
        lease = RefLease(
            ref=ref_index,
            call_uuid=call_uuid,
            lease=self._next_lease[ref_index],
            issued_frame=issued_frame,
            issued_micro=issued_micro,
        )
        self._active[ref_index] = lease
        return lease

    def reserve_parsed(self, action: ParsedAction, *, call_uuid: str) -> RefLease:
        if not action.dispatchable or action.ref is None:
            raise RefLeaseError("only a parsed CALL_END action can reserve a V2 REF")
        return self.reserve(
            action.ref,
            call_uuid=call_uuid,
            issued_frame=action.issued_frame,
            issued_micro=action.issued_micro,
        )

    def validate_event(self, ref: int | str, *, call_uuid: str, lease: int) -> RefLease:
        """Return the active lease or reject a stale/misattributed tool event."""

        if not isinstance(call_uuid, str) or not call_uuid:
            raise RefLeaseError("tool event requires the hidden call_uuid")
        if isinstance(lease, bool) or not isinstance(lease, int) or lease < 1:
            raise RefLeaseError("tool event lease must be a positive integer")
        current = self.lease_for(ref)
        if current.call_uuid != call_uuid or current.lease != lease:
            raise RefLeaseError(f"REF_{current.ref} event belongs to a stale or different lease")
        return current

    def release_terminal(self, receipt: TerminalLeaseReceipt) -> RefLease:
        """Release after the scheduler records a matching final ``OBS_END``."""

        if not isinstance(receipt, TerminalLeaseReceipt):
            raise RefLeaseError("REF release requires a scheduler terminal OBS_END receipt")
        current = self.validate_event(
            receipt.ref,
            call_uuid=receipt.call_uuid,
            lease=receipt.lease,
        )
        if receipt.available_frame <= current.issued_frame:
            raise RefLeaseError("terminal observation must be available strictly after CALL_END")
        # Given available >= issued + 1 and injected_end >= available, the
        # one-step input shift makes visible_frame at least issued + 2.
        if receipt.visible_frame <= current.issued_frame + 1:
            raise RefLeaseError("terminal observation is not causally visible after CALL_END")
        del self._active[current.ref]
        return current

    # A concise alias for runtime adapters that name this operation ``release``.
    release = release_terminal


@dataclass(frozen=True, slots=True)
class MicroLane:
    """Materialized [frame, micro-slot] action IDs plus an explicit loss mask."""

    ids: tuple[tuple[int, ...], ...]
    mask: tuple[tuple[bool, ...], ...]
    padding_id: int = PAD_TOKEN_ID

    def __post_init__(self) -> None:
        if self.padding_id != PAD_TOKEN_ID:
            raise ValueError("V2 action lane padding is fixed at PAD=-1")
        ids = tuple(tuple(frame) for frame in self.ids)
        mask = tuple(tuple(frame) for frame in self.mask)
        if len(ids) != len(mask):
            raise ValueError("V2 action lane ids and mask must have equal frame counts")
        for frame_ids, frame_mask in zip(ids, mask, strict=True):
            if len(frame_ids) != MICRO_TOKENS_PER_FRAME or len(frame_mask) != MICRO_TOKENS_PER_FRAME:
                raise ValueError("every V2 action lane frame has exactly five micro-slots")
            padding_started = False
            for token_id, scored in zip(frame_ids, frame_mask, strict=True):
                if not isinstance(scored, bool):
                    raise ValueError("V2 action lane mask values must be booleans")
                if token_id == self.padding_id:
                    padding_started = True
                    if scored:
                        raise ValueError("PAD=-1 must never be an action target")
                else:
                    if (
                        isinstance(token_id, bool)
                        or not isinstance(token_id, int)
                        or token_id < 0
                        or token_id >= 256
                    ):
                        raise ValueError("V2 action lane IDs must be in the fixed 0..255 range")
                    if not scored:
                        raise ValueError("every semantic action token must be a supervised target")
                    if padding_started:
                        raise ValueError("V2 action PAD must be a contiguous frame suffix")
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "mask", mask)

    @property
    def frame_count(self) -> int:
        return len(self.ids)

    @property
    def action_ids(self) -> tuple[tuple[int, ...], ...]:
        return self.ids

    @property
    def action_loss_mask(self) -> tuple[tuple[bool, ...], ...]:
        return self.mask

    def visible_tokens(self, frame: int) -> tuple[int, ...]:
        return tuple(
            token_id
            for token_id, scored in zip(self.ids[frame], self.mask[frame], strict=True)
            if scored
        )

    def to_dict(self) -> dict[str, object]:
        return {"ids": [list(frame) for frame in self.ids], "mask": [list(frame) for frame in self.mask]}


def materialize_action_lane(
    frame_count: int,
    packets: Iterable[ActionPacketV2],
    *,
    manifest: ActionV2Manifest = DEFAULT_MANIFEST,
    active_refs: object | None = None,
) -> MicroLane:
    """Materialize valid V2 packets into a fixed [T, 5] target lane.

    Every 80 ms frame has an explicit supervised ``NOOP`` at micro-slot zero
    when no packet occupies that logical two-frame region.  A packet reserves
    both physical frames even when its semantic sequence ends in the first;
    this keeps the 160 ms packet clock unambiguous and detects timeline-edge
    mistakes early.  This function is a tensor renderer, not a runtime lease
    timeline: it can check an initial ``active_refs`` admission set but cannot
    reserve/release refs because UUIDs and terminal receipts are intentionally
    absent from model target packets.  Use :class:`RefLeaseGuard` for the
    chronological runtime/corpus validation path.
    """

    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
        raise ValueError("frame_count must be a positive integer")
    parser = ActionParser(manifest)
    defaults = [manifest.action_id("NOOP")] + [PAD_TOKEN_ID] * (MICRO_TOKENS_PER_FRAME - 1)
    ids = [list(defaults) for _ in range(frame_count)]
    mask = [[True] + [False] * (MICRO_TOKENS_PER_FRAME - 1) for _ in range(frame_count)]
    occupied: set[int] = set()

    packet_values = tuple(packets)
    if not all(isinstance(packet, ActionPacketV2) for packet in packet_values):
        raise ValueError("materialize_action_lane accepts only ActionPacketV2 values")
    ordered_packets = sorted(packet_values, key=lambda packet: packet.packet_start_frame)
    runtime_active = active_refs
    for packet in ordered_packets:
        parser.parse(packet, active_refs=runtime_active)
        packet_end_exclusive = packet.packet_start_frame + PACKET_FRAMES
        if packet_end_exclusive > frame_count:
            raise ValueError("V2 action packet's full two-frame region exceeds the timeline")
        physical_frames = range(packet.packet_start_frame, packet_end_exclusive)
        overlap = set(physical_frames) & occupied
        if overlap:
            raise ValueError(f"V2 action packets collide in frame(s) {sorted(overlap)}")
        occupied.update(physical_frames)
        for token_index, token_id in enumerate(packet.token_ids):
            frame = packet.packet_start_frame + (token_index // MICRO_TOKENS_PER_FRAME)
            micro = token_index % MICRO_TOKENS_PER_FRAME
            ids[frame][micro] = token_id
            mask[frame][micro] = True
        # Pad only the suffix of frames to which this packet actually wrote
        # semantic tokens.  A completely unused second frame remains the
        # normal [NOOP, PAD, ...] target for that 80 ms tick.
        for frame in physical_frames:
            available = len(packet.token_ids) - (
                frame - packet.packet_start_frame
            ) * MICRO_TOKENS_PER_FRAME
            if available <= 0:
                continue
            populated = min(MICRO_TOKENS_PER_FRAME, available)
            for micro in range(populated, MICRO_TOKENS_PER_FRAME):
                ids[frame][micro] = PAD_TOKEN_ID
                mask[frame][micro] = False

    return MicroLane(
        ids=tuple(tuple(frame) for frame in ids),
        mask=tuple(tuple(frame) for frame in mask),
    )


# Explicit V2 aliases make call sites self-documenting without adding a second
# implementation.
ActionV2Compiler = ActionCompiler
ActionV2Parser = ActionParser


__all__ = [
    "ACTION_PACKET_FRAMES",
    "ACTION_V2_VERSION",
    "ActionCompiler",
    "ActionGrammarError",
    "ActionKind",
    "ActionPacketV2",
    "ActionParser",
    "ActionProtocolError",
    "ActionV2Compiler",
    "ActionV2Manifest",
    "ActionV2Parser",
    "BASE_FRAME_MS",
    "ControlAction",
    "DEFAULT_MANIFEST",
    "MAX_ACTION_TOKENS_PER_FRAME",
    "MAX_ACTION_TOKENS_PER_PACKET",
    "MAX_REFS",
    "MICRO_TOKENS_PER_FRAME",
    "MicroLane",
    "MicroPosition",
    "PAD_TOKEN_ID",
    "PACKET_FRAMES",
    "ParsedAction",
    "RefLease",
    "RefLeaseError",
    "RefLeaseGuard",
    "TerminalLeaseReceipt",
    "ToolArgumentSpec",
    "ToolSpec",
    "default_manifest",
    "materialize_action_lane",
]
