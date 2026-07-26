"""Deterministic natural-speech scenarios for PersonaPlex-Agent V2 training.

The symbolic corpus proves action/result causality, but its ``grounded_text`` is
protocol metadata rather than something a person should say.  This module adds
an immutable speech manifest whose natural utterances are derived from the
exact fixed-vocabulary arguments and executor results produced by
``SymbolicCorpusBuilder``.

No audio is synthesized here.  The first validation tier holds out speech
templates and audio while reusing symbolic output classes taught during
training.  A fresh fixed-vocabulary head cannot learn an arbitrary ``CITY_n``
or ``MIN_n`` class that never appears as a target.  Broader unseen-value
coverage is therefore a later curriculum stage, not the initial autonomous
call proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

from .action_v2 import DEFAULT_MANIFEST
from .corpus_v2 import ScenarioSpec, SymbolicCorpusBuilder


SPEECH_SCENARIO_CATALOG_VERSION = "ppx-speech-scenarios-v2.2.0"
TEXT_BINDING_VERSION = "ppx-natural-bindings-v1.0.0"


class SpeechScenarioError(ValueError):
    """Raised when natural speech and its symbolic V2 meaning disagree."""


class ScenarioSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"


CITY_NAMES = (
    "Zagreb", "Split", "Rijeka", "Osijek", "Zadar", "Dubrovnik", "Pula", "Varazdin",
    "Sibenik", "Karlovac", "Sisak", "Vukovar", "Samobor", "Trogir", "Makarska", "Koprivnica",
    "London", "Paris", "Berlin", "Madrid", "Rome", "Vienna", "Prague", "Warsaw",
    "Lisbon", "Dublin", "Oslo", "Stockholm", "Helsinki", "Copenhagen", "Brussels", "Amsterdam",
)

FORECAST_PHRASES = (
    "clear skies", "mostly sunny", "partly cloudy", "overcast", "light rain", "steady rain",
    "thunderstorms", "light snow", "heavy snow", "fog", "strong winds", "humid conditions",
    "dry conditions", "scattered showers", "freezing rain", "haze",
)

ROOM_NAMES = (
    "living room", "bedroom", "kitchen", "office", "nursery", "guest room",
    "dining room", "basement",
)

_TOOLS = tuple(tool.name for tool in DEFAULT_MANIFEST.tool_schemas)
_RAW_PROTOCOL_MARKERS = (
    "CITY_", "MIN_", "ROOM_", "TEMP_C_", "FORECAST_", "ENUM_", "TOOL_", "REF_",
)

_TRAIN_NO_TOOL = (
    ("greeting", "Hello, how are you today?", "I'm doing well, thanks. How are you?"),
    ("clarification", "What did you mean by that last point?", "I meant that we should verify the details before deciding."),
    ("empathy", "I had a really difficult day.", "I'm sorry it was so difficult. Do you want to talk about what happened?"),
    ("opinion", "Which season do you enjoy most?", "I enjoy spring because everything feels fresh and active again."),
    ("story", "Tell me a short story about a lighthouse.", "A lighthouse keeper once guided a lost boat home through a silent winter storm."),
    ("arithmetic", "What is seven plus five?", "Seven plus five is twelve."),
    ("tool_adjacent_timer", "I watched a movie called The Timer last night.", "That title sounds suspenseful. Did you enjoy the movie?"),
    ("tool_adjacent_temperature", "Twenty degrees feels comfortable to me.", "That sounds like a comfortable temperature for many people."),
    ("tool_adjacent_weather_report", "Weather reports can be surprisingly dramatic to read.", "They can. The wording often makes ordinary changes sound quite vivid."),
    ("tool_adjacent_weather_preference", "I prefer sunny weather to rainy weather.", "Sunny days can feel brighter and make outdoor plans easier."),
    ("tool_adjacent_forecast_word", "The word forecast sounds more formal than prediction.", "It does sound more specialized, especially when people discuss weather or planning."),
    ("tool_adjacent_city_climate", "My friend moved to Makarska because she enjoys the climate.", "Makarska has a coastal setting that many people find appealing."),
    ("tool_adjacent_timer_story", "The kitchen timer was a funny plot device in that movie.", "That sounds like a clever way to build tension or comedy into the scene."),
    ("tool_adjacent_timer_past", "I finished the task before the timer rang.", "Finishing ahead of the timer must have felt satisfying."),
    ("tool_adjacent_time_opinion", "Time seems to pass faster when I am busy.", "Many people notice that because focused attention leaves less room to track each minute."),
    ("tool_adjacent_duration_music", "A two minute song can feel surprisingly short.", "It can, especially when the melody is engaging and ends quickly."),
    ("tool_adjacent_living_room", "The living room in my old house was painted blue.", "Blue can give a living room a calm and memorable atmosphere."),
    ("tool_adjacent_room_temperature", "A room around twenty two degrees usually feels comfortable to me.", "That is a comfortable indoor temperature for many people."),
    ("tool_adjacent_bedroom", "My bedroom gets a lot of sunlight in the morning.", "Morning sunlight can make a bedroom feel warm and welcoming."),
    ("tool_adjacent_thermostat", "A thermostat is an interesting example of feedback control.", "Yes, it measures conditions and adjusts the system toward a target."),
)

_VALIDATION_NO_TOOL = (
    ("greeting", "Good evening, it's nice to meet you.", "Good evening. It's nice to meet you too."),
    ("clarification", "Could you explain your previous sentence another way?", "Of course. I'll restate it more simply."),
    ("empathy", "I'm nervous about tomorrow.", "That makes sense. We can talk through what is worrying you."),
    ("tool_adjacent_weather", "Weather is my favorite conversation topic.", "It can be a great topic because it affects everyone's day."),
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _token_number(token: str, prefix: str, upper_exclusive: int) -> int:
    if not isinstance(token, str) or not token.startswith(prefix):
        raise SpeechScenarioError(f"expected {prefix} symbolic token, got {token!r}")
    try:
        value = int(token.removeprefix(prefix))
    except ValueError as exc:
        raise SpeechScenarioError(f"invalid numeric symbolic token {token!r}") from exc
    if value < 0 or value >= upper_exclusive:
        raise SpeechScenarioError(f"symbolic token {token!r} is outside its natural binding")
    return value


def _duration_phrase(minutes: int) -> str:
    return "1 minute" if minutes == 1 else f"{minutes} minutes"


def _natural_bindings(
    tool_name: str,
    arguments: Mapping[str, str],
    payload_tokens: tuple[str, ...],
) -> tuple[dict[str, str], dict[str, str]]:
    """Return exact input/result token-to-speech bindings for one successful call."""

    if tool_name == "weather.lookup":
        if tuple(arguments) != ("city",):
            raise SpeechScenarioError("weather speech binding requires exactly the city argument")
        if len(payload_tokens) != 2 or not payload_tokens[0].startswith("FORECAST_") or not payload_tokens[1].startswith("TEMP_C_"):
            raise SpeechScenarioError("weather result must be FORECAST_n followed by TEMP_C_n")
        city = CITY_NAMES[_token_number(arguments["city"], "CITY_", len(CITY_NAMES))]
        forecast = FORECAST_PHRASES[
            _token_number(payload_tokens[0], "FORECAST_", len(FORECAST_PHRASES))
        ]
        temperature = _token_number(payload_tokens[1], "TEMP_C_", 41)
        if temperature < 10:
            raise SpeechScenarioError("weather temperature binding is outside TEMP_C_10..40")
        return (
            {arguments["city"]: city},
            {
                payload_tokens[0]: forecast,
                payload_tokens[1]: f"{temperature} degrees Celsius",
            },
        )
    if tool_name == "timer.create":
        if tuple(arguments) != ("minutes",) or payload_tokens != ("ENUM_0",):
            raise SpeechScenarioError("timer binding requires MIN_1..60 and exactly ENUM_0")
        minutes = _token_number(arguments["minutes"], "MIN_", 61)
        if minutes < 1:
            raise SpeechScenarioError("timer duration must be at least one minute")
        return ({arguments["minutes"]: _duration_phrase(minutes)}, {"ENUM_0": "timer is set"})
    if tool_name == "home.set_temperature":
        if tuple(arguments) != ("room", "temperature_c") or payload_tokens != ("ENUM_1",):
            raise SpeechScenarioError(
                "home binding requires ordered room/temperature arguments and exactly ENUM_1"
            )
        room = ROOM_NAMES[_token_number(arguments["room"], "ROOM_", len(ROOM_NAMES))]
        temperature = _token_number(arguments["temperature_c"], "TEMP_C_", 41)
        if temperature < 10:
            raise SpeechScenarioError("home temperature binding is outside TEMP_C_10..40")
        return (
            {
                arguments["room"]: room,
                arguments["temperature_c"]: f"{temperature} degrees Celsius",
            },
            {"ENUM_1": "temperature is set"},
        )
    raise SpeechScenarioError(f"unsupported V2 speech tool {tool_name!r}")


def _assert_natural_text(text: str, *, field: str) -> None:
    if not isinstance(text, str) or not text or text != text.strip():
        raise SpeechScenarioError(f"{field} must be non-empty natural text without outer whitespace")
    if any(marker in text for marker in _RAW_PROTOCOL_MARKERS):
        raise SpeechScenarioError(f"{field} exposes a raw V2 protocol token")
    if any(ord(character) < 32 and character not in "\t\n" for character in text):
        raise SpeechScenarioError(f"{field} contains a control character")


@dataclass(frozen=True, slots=True)
class SpeechScenarioV2:
    scenario_id: str
    split: ScenarioSplit
    split_group: str
    template_id: str
    user_utterance: str
    assistant_response: str
    assistant_pending: str | None = None
    tool_name: str | None = None
    arguments: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    terminal_payload_tokens: tuple[str, ...] = ()
    corpus_spec: ScenarioSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.split, ScenarioSplit):
            raise SpeechScenarioError("scenario split must be train or validation")
        expected_prefix = f"SCV2-{self.split.value.upper()}-"
        if not self.scenario_id.startswith(expected_prefix):
            raise SpeechScenarioError("scenario ID does not match its explicit split")
        if not self.split_group.startswith(f"{self.split.value}:"):
            raise SpeechScenarioError("split_group must be explicitly namespaced by split")
        if not self.template_id.startswith(f"{self.split.value}:"):
            raise SpeechScenarioError("template_id must be explicitly namespaced by split")
        _assert_natural_text(self.user_utterance, field="user_utterance")
        _assert_natural_text(self.assistant_response, field="assistant_response")
        if self.assistant_pending is not None:
            _assert_natural_text(self.assistant_pending, field="assistant_pending")

        arguments = dict(self.arguments)
        payload = tuple(self.terminal_payload_tokens)
        object.__setattr__(self, "arguments", MappingProxyType(arguments))
        object.__setattr__(self, "terminal_payload_tokens", payload)

        if self.tool_name is None:
            if arguments or payload or self.corpus_spec is not None or self.assistant_pending is not None:
                raise SpeechScenarioError(
                    "no-tool speech scenarios cannot contain calls, results, specs, or pending speech"
                )
            return

        if self.tool_name not in _TOOLS or self.corpus_spec is None:
            raise SpeechScenarioError("tool speech scenario requires a fixed V2 tool and corpus spec")
        tool = DEFAULT_MANIFEST.tool(self.tool_name)
        expected_arguments = tuple(argument.name for argument in tool.arguments)
        if tuple(arguments) != expected_arguments:
            raise SpeechScenarioError("speech arguments must preserve fixed positional schema order")
        for argument in tool.arguments:
            if arguments[argument.name] not in argument.allowed_tokens:
                raise SpeechScenarioError("speech argument is outside the fixed V2 vocabulary")
        if self.assistant_pending is None:
            raise SpeechScenarioError("tool scenario requires a natural pending acknowledgement")
        input_bindings, result_bindings = _natural_bindings(self.tool_name, arguments, payload)
        user_folded = self.user_utterance.casefold()
        response_folded = self.assistant_response.casefold()
        for phrase in input_bindings.values():
            if phrase.casefold() not in user_folded:
                raise SpeechScenarioError("user utterance does not express its symbolic arguments")
        for phrase in result_bindings.values():
            if phrase.casefold() not in response_folded:
                raise SpeechScenarioError("assistant response does not express its symbolic result")

    @property
    def is_tool_call(self) -> bool:
        return self.tool_name is not None

    @property
    def input_bindings(self) -> Mapping[str, str]:
        if self.tool_name is None:
            return MappingProxyType({})
        inputs, _ = _natural_bindings(
            self.tool_name, self.arguments, self.terminal_payload_tokens
        )
        return MappingProxyType(inputs)

    @property
    def result_bindings(self) -> Mapping[str, str]:
        if self.tool_name is None:
            return MappingProxyType({})
        _, results = _natural_bindings(
            self.tool_name, self.arguments, self.terminal_payload_tokens
        )
        return MappingProxyType(results)

    @property
    def semantic_key(self) -> str:
        if self.tool_name is None:
            value: object = {"no_tool_group": self.split_group}
        else:
            value = {
                "tool_name": self.tool_name,
                "arguments": dict(self.arguments),
                "terminal_payload_tokens": list(self.terminal_payload_tokens),
            }
        return _fingerprint(value)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "split": self.split.value,
            "split_group": self.split_group,
            "template_id": self.template_id,
            "user_utterance": self.user_utterance,
            "assistant_pending": self.assistant_pending,
            "assistant_response": self.assistant_response,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "terminal_payload_tokens": list(self.terminal_payload_tokens),
            "input_bindings": dict(self.input_bindings),
            "result_bindings": dict(self.result_bindings),
            "corpus_spec": None if self.corpus_spec is None else self.corpus_spec.to_dict(),
            "text_binding_version": TEXT_BINDING_VERSION,
        }


def _arguments_belong_to_split(
    split: ScenarioSplit, tool_name: str, arguments: Mapping[str, str]
) -> bool:
    if tool_name == "weather.lookup":
        city = _token_number(arguments["city"], "CITY_", 32)
        return city < 24 if split is ScenarioSplit.TRAIN else city >= 24
    if tool_name == "timer.create":
        minutes = _token_number(arguments["minutes"], "MIN_", 61)
        return minutes <= 45 if split is ScenarioSplit.TRAIN else minutes >= 46
    room = _token_number(arguments["room"], "ROOM_", 8)
    temperature = _token_number(arguments["temperature_c"], "TEMP_C_", 41)
    if split is ScenarioSplit.TRAIN:
        return room <= 5 and temperature <= 30
    return room >= 6 and temperature >= 31


@dataclass(frozen=True, slots=True)
class SpeechScenarioManifestV2:
    scenarios: tuple[SpeechScenarioV2, ...]
    catalog_version: str = SPEECH_SCENARIO_CATALOG_VERSION
    protocol_version: str = DEFAULT_MANIFEST.version
    manifest_fingerprint: str = DEFAULT_MANIFEST.fingerprint
    tool_catalog_hash: str = DEFAULT_MANIFEST.tool_catalog_hash
    text_binding_version: str = TEXT_BINDING_VERSION

    def __post_init__(self) -> None:
        scenarios = tuple(self.scenarios)
        if not scenarios or not all(isinstance(item, SpeechScenarioV2) for item in scenarios):
            raise SpeechScenarioError("speech manifest requires immutable SpeechScenarioV2 rows")
        if self.catalog_version != SPEECH_SCENARIO_CATALOG_VERSION:
            raise SpeechScenarioError("speech catalog version drifted")
        if (
            self.protocol_version != DEFAULT_MANIFEST.version
            or self.manifest_fingerprint != DEFAULT_MANIFEST.fingerprint
            or self.tool_catalog_hash != DEFAULT_MANIFEST.tool_catalog_hash
            or self.text_binding_version != TEXT_BINDING_VERSION
        ):
            raise SpeechScenarioError("speech manifest is not pinned to the fixed V2 protocol")
        ids = [item.scenario_id for item in scenarios]
        if len(ids) != len(set(ids)):
            raise SpeechScenarioError("scenario IDs must be globally unique")
        group_splits: dict[str, ScenarioSplit] = {}
        for item in scenarios:
            previous = group_splits.setdefault(item.split_group, item.split)
            if previous is not item.split:
                raise SpeechScenarioError("one split_group cannot cross train/validation")
        for split in ScenarioSplit:
            subset = tuple(item for item in scenarios if item.split is split)
            coverage = {item.tool_name for item in subset}
            if coverage != {None, *_TOOLS}:
                raise SpeechScenarioError(f"{split.value} split lacks required tool/no-tool coverage")
            for item in subset:
                if (
                    split is ScenarioSplit.TRAIN
                    and item.tool_name is not None
                    and not _arguments_belong_to_split(split, item.tool_name, item.arguments)
                ):
                    raise SpeechScenarioError("training argument left its pinned starter domain")
        train_tokens_by_tool = {
            tool_name: {
                token
                for item in scenarios
                if item.split is ScenarioSplit.TRAIN and item.tool_name == tool_name
                for token in item.arguments.values()
            }
            for tool_name in _TOOLS
        }
        for item in scenarios:
            if item.split is ScenarioSplit.VALIDATION and item.tool_name is not None:
                if not set(item.arguments.values()) <= train_tokens_by_tool[item.tool_name]:
                    raise SpeechScenarioError(
                        "validation uses an output class absent from training"
                    )
        builder = SymbolicCorpusBuilder()
        for item in scenarios:
            if item.tool_name is None:
                continue
            assert item.corpus_spec is not None
            record = builder.build(item.corpus_spec)
            execution = record.executions[0]
            if (
                execution.parsed_action.tool_name != item.tool_name
                or dict(execution.parsed_action.arguments) != dict(item.arguments)
                or execution.outcome.payload_tokens != item.terminal_payload_tokens
            ):
                raise SpeechScenarioError(
                    "natural speech row drifted from its deterministic symbolic corpus spec"
                )
        object.__setattr__(self, "scenarios", scenarios)

    def for_split(self, split: ScenarioSplit | str) -> tuple[SpeechScenarioV2, ...]:
        try:
            normalized = split if isinstance(split, ScenarioSplit) else ScenarioSplit(split)
        except ValueError as exc:
            raise SpeechScenarioError(f"unknown speech scenario split {split!r}") from exc
        return tuple(item for item in self.scenarios if item.split is normalized)

    def _body(self) -> dict[str, object]:
        return {
            "catalog_version": self.catalog_version,
            "protocol_version": self.protocol_version,
            "manifest_fingerprint": self.manifest_fingerprint,
            "tool_catalog_hash": self.tool_catalog_hash,
            "text_binding_version": self.text_binding_version,
            "scenario_count": len(self.scenarios),
            "scenarios": [item.to_dict() for item in self.scenarios],
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self._body())

    def to_dict(self) -> dict[str, object]:
        body = self._body()
        body["catalog_fingerprint"] = self.fingerprint
        return body


_USER_TEMPLATES = {
    (ScenarioSplit.TRAIN, "weather.lookup"): (
        "What's the weather in {city}?",
        "Could you check the forecast for {city}?",
        "How is the weather looking in {city} today?",
    ),
    (ScenarioSplit.VALIDATION, "weather.lookup"): (
        "Please look up today's conditions in {city}.",
        "What conditions should I expect in {city}?",
    ),
    (ScenarioSplit.TRAIN, "timer.create"): (
        "Set a timer for {duration}.",
        "Please start a {duration} timer.",
        "Can you create a timer lasting {duration}?",
    ),
    (ScenarioSplit.VALIDATION, "timer.create"): (
        "Count down {duration} for me.",
        "I'd like an alarm after {duration}.",
    ),
    (ScenarioSplit.TRAIN, "home.set_temperature"): (
        "Set the {room} to {temperature}.",
        "Please change the {room} temperature to {temperature}.",
        "Make it {temperature} in the {room}.",
    ),
    (ScenarioSplit.VALIDATION, "home.set_temperature"): (
        "Adjust the {room} so it reaches {temperature}.",
        "I want the {room} kept at {temperature}.",
    ),
}


class SpeechScenarioFactoryV2:
    """Build the pinned multi-example catalog without randomness or I/O."""

    def __init__(self, *, base_seed: int = 20260710) -> None:
        if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
            raise SpeechScenarioError("speech scenario base_seed must be non-negative")
        self.base_seed = base_seed
        self._builder = SymbolicCorpusBuilder()

    def build_default(self) -> SpeechScenarioManifestV2:
        scenarios: list[SpeechScenarioV2] = []
        training_by_tool: dict[str, tuple[SpeechScenarioV2, ...]] = {}
        for tool_name in _TOOLS:
            rows = self._tool_rows(ScenarioSplit.TRAIN, tool_name, 12)
            training_by_tool[tool_name] = rows
            scenarios.extend(rows)
        scenarios.extend(self._no_tool_rows(ScenarioSplit.TRAIN, _TRAIN_NO_TOOL))

        for tool_name in _TOOLS:
            scenarios.extend(
                self._held_out_phrase_rows(tool_name, training_by_tool[tool_name])
            )
        scenarios.extend(
            self._no_tool_rows(ScenarioSplit.VALIDATION, _VALIDATION_NO_TOOL)
        )
        return SpeechScenarioManifestV2(tuple(scenarios))

    def _held_out_phrase_rows(
        self,
        tool_name: str,
        training_rows: tuple[SpeechScenarioV2, ...],
    ) -> tuple[SpeechScenarioV2, ...]:
        """Render unseen validation phrasing for four learned output classes."""

        if len(training_rows) != 12:
            raise SpeechScenarioError("held-out phrasing requires twelve training rows")
        selected = tuple(training_rows[index] for index in (0, 3, 6, 9))
        result: list[SpeechScenarioV2] = []
        for ordinal, source in enumerate(selected, start=1):
            if source.corpus_spec is None:
                raise SpeechScenarioError("tool validation source is missing its corpus spec")
            result.append(
                self._render_tool_row(
                    ScenarioSplit.VALIDATION,
                    tool_name,
                    ordinal,
                    source.corpus_spec,
                    source.arguments,
                    source.terminal_payload_tokens,
                )
            )
        return tuple(result)

    def _tool_rows(
        self, split: ScenarioSplit, tool_name: str, count: int
    ) -> tuple[SpeechScenarioV2, ...]:
        tool_index = _TOOLS.index(tool_name)
        split_offset = 0 if split is ScenarioSplit.TRAIN else 1_000_000
        seed_start = self.base_seed + split_offset + tool_index * 100_000
        selected: list[SpeechScenarioV2] = []
        seen_arguments: set[tuple[tuple[str, str], ...]] = set()
        for attempt in range(20_000):
            spec = ScenarioSpec(
                seed=seed_start + attempt,
                frame_count=20,
                call_count=1,
                first_packet_frame=2,
                packet_stride_frames=2,
                pending_delay_frames=1,
                terminal_latency_frames=(3,),
                tool_sequence=(tool_name,),
            )
            record = self._builder.build(spec)
            execution = record.executions[0]
            arguments = dict(execution.parsed_action.arguments)
            argument_key = tuple(arguments.items())
            if (
                argument_key in seen_arguments
                or not _arguments_belong_to_split(split, tool_name, arguments)
            ):
                continue
            seen_arguments.add(argument_key)
            ordinal = len(selected) + 1
            selected.append(
                self._render_tool_row(
                    split,
                    tool_name,
                    ordinal,
                    spec,
                    arguments,
                    execution.outcome.payload_tokens,
                )
            )
            if len(selected) == count:
                return tuple(selected)
        raise SpeechScenarioError(f"could not fill deterministic {split.value} {tool_name} rows")

    @staticmethod
    def _render_tool_row(
        split: ScenarioSplit,
        tool_name: str,
        ordinal: int,
        spec: ScenarioSpec,
        arguments: Mapping[str, str],
        payload_tokens: tuple[str, ...],
    ) -> SpeechScenarioV2:
        inputs, results = _natural_bindings(tool_name, arguments, payload_tokens)
        templates = _USER_TEMPLATES[(split, tool_name)]
        template_index = (ordinal - 1) % len(templates)
        if tool_name == "weather.lookup":
            city = inputs[arguments["city"]]
            user = templates[template_index].format(city=city)
            pending = f"I'll check {city} now."
            response = (
                f"In {city}, the forecast is {results[payload_tokens[0]]}, with "
                f"{results[payload_tokens[1]]}."
            )
            slug = "WEATHER"
        elif tool_name == "timer.create":
            duration = inputs[arguments["minutes"]]
            user = templates[template_index].format(duration=duration)
            pending = f"I'll start the {duration} timer now."
            response = f"Your {duration} timer is set."
            slug = "TIMER"
        else:
            room = inputs[arguments["room"]]
            temperature = inputs[arguments["temperature_c"]]
            user = templates[template_index].format(room=room, temperature=temperature)
            pending = f"I'll update the {room} temperature."
            response = f"The {room} temperature is set to {temperature}."
            slug = "HOME"
        number = f"{ordinal:03d}"
        return SpeechScenarioV2(
            scenario_id=f"SCV2-{split.value.upper()}-{slug}-{number}",
            split=split,
            split_group=f"{split.value}:{tool_name}:{spec.fingerprint}",
            template_id=f"{split.value}:{tool_name}:template-{template_index}",
            user_utterance=user,
            assistant_pending=pending,
            assistant_response=response,
            tool_name=tool_name,
            arguments=arguments,
            terminal_payload_tokens=payload_tokens,
            corpus_spec=spec,
        )

    @staticmethod
    def _no_tool_rows(
        split: ScenarioSplit,
        rows: tuple[tuple[str, str, str], ...],
    ) -> tuple[SpeechScenarioV2, ...]:
        result: list[SpeechScenarioV2] = []
        for index, (family, user, response) in enumerate(rows, start=1):
            result.append(
                SpeechScenarioV2(
                    scenario_id=(
                        f"SCV2-{split.value.upper()}-NO-TOOL-{index:03d}"
                    ),
                    split=split,
                    split_group=f"{split.value}:no_tool:{family}:{index}",
                    template_id=f"{split.value}:no_tool:{family}",
                    user_utterance=user,
                    assistant_response=response,
                )
            )
        return tuple(result)


def build_default_speech_scenario_manifest() -> SpeechScenarioManifestV2:
    return SpeechScenarioFactoryV2().build_default()


__all__ = [
    "CITY_NAMES",
    "FORECAST_PHRASES",
    "ROOM_NAMES",
    "SPEECH_SCENARIO_CATALOG_VERSION",
    "TEXT_BINDING_VERSION",
    "ScenarioSplit",
    "SpeechScenarioError",
    "SpeechScenarioFactoryV2",
    "SpeechScenarioManifestV2",
    "SpeechScenarioV2",
    "build_default_speech_scenario_manifest",
]
