"""Validated PersonaPlex base-stream assembly for aligned V2 training records."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Sequence


TEXT_STREAM_INDEX = 0
ASSISTANT_AUDIO_START = 1
USER_AUDIO_START = 9
AUDIO_CODEBOOKS_PER_SPEAKER = 8
PERSONAPLEX_BASE_STREAMS = 17
DEFAULT_TEXT_PADDING_ID = 3
DEFAULT_TEXT_CARDINALITY = 32_000
DEFAULT_AUDIO_CARDINALITY = 2_048


class BaseCodesV2Error(ValueError):
    """Raised when a materialized PersonaPlex base-stream matrix is invalid."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _validate_int(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BaseCodesV2Error(f"{name} must be an integer >= {minimum}")
    return value


def _normalize_audio(
    name: str,
    codes: Iterable[Iterable[int]],
    *,
    audio_cardinality: int,
) -> tuple[tuple[int, ...], ...]:
    normalized = tuple(tuple(stream) for stream in codes)
    if len(normalized) != AUDIO_CODEBOOKS_PER_SPEAKER:
        raise BaseCodesV2Error(f"{name} must contain exactly eight Mimi codebooks")
    lengths = {len(stream) for stream in normalized}
    if len(lengths) != 1:
        raise BaseCodesV2Error(f"{name} Mimi codebooks must have equal frame counts")
    for stream in normalized:
        for token in stream:
            if (
                isinstance(token, bool)
                or not isinstance(token, int)
                or token < 0
                or token >= audio_cardinality
            ):
                raise BaseCodesV2Error(
                    f"{name} contains an ID outside 0..{audio_cardinality - 1}"
                )
    return normalized


@dataclass(frozen=True, slots=True)
class PersonaPlexBaseCodes:
    """Materialized `[17,T]` PersonaPlex codes and their speech spans.

    Stream 0 is assistant speech-anchor text, streams 1..8 are assistant
    audio targets, and streams 9..16 are observed user audio.  This class is
    model data only; action and executor lanes remain separate `[T,5]` arrays.
    """

    codes: tuple[tuple[int, ...], ...]
    user_audio_span: tuple[int, int]
    assistant_audio_span: tuple[int, int] | None
    text_padding_id: int = DEFAULT_TEXT_PADDING_ID
    text_cardinality: int = DEFAULT_TEXT_CARDINALITY
    audio_cardinality: int = DEFAULT_AUDIO_CARDINALITY
    encoding_mode: str = "causal_streaming_80ms"

    def __post_init__(self) -> None:
        streams = tuple(tuple(stream) for stream in self.codes)
        if len(streams) != PERSONAPLEX_BASE_STREAMS:
            raise BaseCodesV2Error("PersonaPlex base codes must have exactly 17 streams")
        lengths = {len(stream) for stream in streams}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
            raise BaseCodesV2Error("all PersonaPlex streams must share a positive frame count")
        _validate_int("text padding ID", self.text_padding_id)
        _validate_int("text cardinality", self.text_cardinality, minimum=1)
        _validate_int("audio cardinality", self.audio_cardinality, minimum=1)
        if self.text_padding_id >= self.text_cardinality:
            raise BaseCodesV2Error("text padding ID is outside the text vocabulary")
        for token in streams[TEXT_STREAM_INDEX]:
            if (
                isinstance(token, bool)
                or not isinstance(token, int)
                or token < 0
                or token >= self.text_cardinality
            ):
                raise BaseCodesV2Error("text stream contains an out-of-range token")
        for stream in streams[1:]:
            for token in stream:
                if (
                    isinstance(token, bool)
                    or not isinstance(token, int)
                    or token < 0
                    or token >= self.audio_cardinality
                ):
                    raise BaseCodesV2Error("audio stream contains an out-of-range Mimi code")
        self._validate_span("user audio span", self.user_audio_span, next(iter(lengths)))
        if self.assistant_audio_span is not None:
            self._validate_span(
                "assistant audio span",
                self.assistant_audio_span,
                next(iter(lengths)),
            )
        if self.encoding_mode != "causal_streaming_80ms":
            raise BaseCodesV2Error("training codes must use live-equivalent causal Mimi encoding")
        object.__setattr__(self, "codes", streams)

    @staticmethod
    def _validate_span(name: str, span: tuple[int, int], frame_count: int) -> None:
        if not isinstance(span, tuple) or len(span) != 2:
            raise BaseCodesV2Error(f"{name} must be a (start, end) tuple")
        start = _validate_int(f"{name} start", span[0])
        end = _validate_int(f"{name} end", span[1])
        if end < start or end > frame_count:
            raise BaseCodesV2Error(f"{name} is outside the aligned timeline")

    @property
    def frame_count(self) -> int:
        return len(self.codes[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (PERSONAPLEX_BASE_STREAMS, self.frame_count)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.to_dict())).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "codes": [list(stream) for stream in self.codes],
            "stream_layout": {
                "assistant_text": 0,
                "assistant_audio": [1, 9],
                "user_audio": [9, 17],
            },
            "user_audio_span": list(self.user_audio_span),
            "assistant_audio_span": (
                None if self.assistant_audio_span is None else list(self.assistant_audio_span)
            ),
            "text_padding_id": self.text_padding_id,
            "text_cardinality": self.text_cardinality,
            "audio_cardinality": self.audio_cardinality,
            "encoding_mode": self.encoding_mode,
        }


def assemble_personaplex_base_codes(
    frame_count: int,
    *,
    silence_tokens: Sequence[int],
    user_audio_codes: Iterable[Iterable[int]],
    user_start_frame: int = 0,
    assistant_audio_codes: Iterable[Iterable[int]] | None = None,
    assistant_start_frame: int = 0,
    text_tokens: Sequence[int] | None = None,
    text_padding_id: int = DEFAULT_TEXT_PADDING_ID,
    text_cardinality: int = DEFAULT_TEXT_CARDINALITY,
    audio_cardinality: int = DEFAULT_AUDIO_CARDINALITY,
) -> PersonaPlexBaseCodes:
    """Place real Mimi codes into the fixed 17-stream PersonaPlex layout."""

    frames = _validate_int("frame_count", frame_count, minimum=1)
    user_start = _validate_int("user_start_frame", user_start_frame)
    assistant_start = _validate_int("assistant_start_frame", assistant_start_frame)
    if len(silence_tokens) != AUDIO_CODEBOOKS_PER_SPEAKER:
        raise BaseCodesV2Error("silence_tokens must contain one real ID per Mimi codebook")
    silence = tuple(silence_tokens)
    if any(
        isinstance(token, bool)
        or not isinstance(token, int)
        or token < 0
        or token >= audio_cardinality
        for token in silence
    ):
        raise BaseCodesV2Error("silence token is outside the Mimi vocabulary")

    user = _normalize_audio(
        "user audio",
        user_audio_codes,
        audio_cardinality=audio_cardinality,
    )
    user_length = len(user[0])
    if user_start + user_length > frames:
        raise BaseCodesV2Error("user audio exceeds the aligned timeline")

    assistant: tuple[tuple[int, ...], ...] | None = None
    assistant_length = 0
    if assistant_audio_codes is not None:
        assistant = _normalize_audio(
            "assistant audio",
            assistant_audio_codes,
            audio_cardinality=audio_cardinality,
        )
        assistant_length = len(assistant[0])
        if assistant_start + assistant_length > frames:
            raise BaseCodesV2Error("assistant audio exceeds the aligned timeline")

    if text_tokens is None:
        text = [text_padding_id] * frames
    else:
        text = list(text_tokens)
        if len(text) != frames:
            raise BaseCodesV2Error("text_tokens must contain exactly one value per frame")

    streams = [text]
    streams.extend([[silence[index]] * frames for index in range(8)])
    streams.extend([[silence[index]] * frames for index in range(8)])
    for codebook, values in enumerate(user):
        streams[USER_AUDIO_START + codebook][
            user_start : user_start + user_length
        ] = values
    if assistant is not None:
        for codebook, values in enumerate(assistant):
            streams[ASSISTANT_AUDIO_START + codebook][
                assistant_start : assistant_start + assistant_length
            ] = values

    return PersonaPlexBaseCodes(
        codes=tuple(tuple(stream) for stream in streams),
        user_audio_span=(user_start, user_start + user_length),
        assistant_audio_span=(
            None
            if assistant is None
            else (assistant_start, assistant_start + assistant_length)
        ),
        text_padding_id=text_padding_id,
        text_cardinality=text_cardinality,
        audio_cardinality=audio_cardinality,
    )


__all__ = [
    "ASSISTANT_AUDIO_START",
    "AUDIO_CODEBOOKS_PER_SPEAKER",
    "BaseCodesV2Error",
    "DEFAULT_AUDIO_CARDINALITY",
    "DEFAULT_TEXT_CARDINALITY",
    "DEFAULT_TEXT_PADDING_ID",
    "PERSONAPLEX_BASE_STREAMS",
    "PersonaPlexBaseCodes",
    "TEXT_STREAM_INDEX",
    "USER_AUDIO_START",
    "assemble_personaplex_base_codes",
]
