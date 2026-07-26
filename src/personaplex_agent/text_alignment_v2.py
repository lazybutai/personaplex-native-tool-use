"""Prototype frame alignment for PersonaPlex assistant speech-anchor tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


TEXT_ALIGNMENT_METHOD = "uniform_within_grounded_audio_v0"
SAPI_WORD_ALIGNMENT_METHOD = "sapi_word_timed_v1"
SUPPORTED_TEXT_ALIGNMENT_METHODS = frozenset(
    {TEXT_ALIGNMENT_METHOD, SAPI_WORD_ALIGNMENT_METHOD}
)


class TextAlignmentV2Error(ValueError):
    """Raised when speech-anchor tokens cannot fit their causal audio span."""


@dataclass(frozen=True, slots=True)
class AlignedTextTokensV2:
    ids: tuple[int, ...]
    semantic_tokens: tuple[int, ...]
    token_frames: tuple[int, ...]
    span_start: int
    span_end_exclusive: int
    padding_id: int = 3
    text_cardinality: int = 32_000
    method: str = TEXT_ALIGNMENT_METHOD

    def __post_init__(self) -> None:
        ids = tuple(self.ids)
        semantic = tuple(self.semantic_tokens)
        frames = tuple(self.token_frames)
        if not ids:
            raise TextAlignmentV2Error("aligned text timeline cannot be empty")
        if len(semantic) != len(frames) or not semantic:
            raise TextAlignmentV2Error("aligned text requires matching semantic tokens and frames")
        if self.span_start < 0 or self.span_end_exclusive <= self.span_start:
            raise TextAlignmentV2Error("aligned text span is invalid")
        if self.span_end_exclusive > len(ids):
            raise TextAlignmentV2Error("aligned text span exceeds its frame timeline")
        if frames != tuple(sorted(set(frames))):
            raise TextAlignmentV2Error("speech-anchor token frames must be unique and ordered")
        if any(frame < self.span_start or frame >= self.span_end_exclusive for frame in frames):
            raise TextAlignmentV2Error("speech-anchor token is outside grounded assistant audio")
        if self.padding_id < 0 or self.padding_id >= self.text_cardinality:
            raise TextAlignmentV2Error("text padding ID is outside the vocabulary")
        if any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            or token >= self.text_cardinality
            for token in semantic
        ):
            raise TextAlignmentV2Error("semantic text token is outside the vocabulary")
        expected = [self.padding_id] * len(ids)
        for frame, token in zip(frames, semantic, strict=True):
            expected[frame] = token
        if ids != tuple(expected):
            raise TextAlignmentV2Error("aligned text IDs do not match token/frame metadata")
        if self.method not in SUPPORTED_TEXT_ALIGNMENT_METHODS:
            raise TextAlignmentV2Error("unknown V2 prototype text-alignment method")
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "semantic_tokens", semantic)
        object.__setattr__(self, "token_frames", frames)

    @property
    def frame_count(self) -> int:
        return len(self.ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "frame_count": self.frame_count,
            "span": [self.span_start, self.span_end_exclusive],
            "padding_id": self.padding_id,
            "semantic_tokens": list(self.semantic_tokens),
            "token_frames": list(self.token_frames),
            "constraint": "all token_frames are within grounded assistant audio",
        }


def uniformly_align_text_tokens(
    frame_count: int,
    token_ids: Iterable[int],
    *,
    span_start: int,
    span_end_exclusive: int,
    padding_id: int = 3,
    text_cardinality: int = 32_000,
) -> AlignedTextTokensV2:
    """Distribute tokenizer IDs monotonically over a causally grounded audio span.

    This is a deterministic bootstrap alignment, not word-level forced
    alignment.  It is deliberately versioned `v0` so a later acoustic aligner
    cannot silently change existing corpus fingerprints.
    """

    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
        raise TextAlignmentV2Error("frame_count must be a positive integer")
    if (
        isinstance(span_start, bool)
        or not isinstance(span_start, int)
        or isinstance(span_end_exclusive, bool)
        or not isinstance(span_end_exclusive, int)
        or span_start < 0
        or span_end_exclusive <= span_start
        or span_end_exclusive > frame_count
    ):
        raise TextAlignmentV2Error("text alignment span is outside the frame timeline")
    tokens = tuple(token_ids)
    if not tokens:
        raise TextAlignmentV2Error("at least one semantic text token is required")
    duration = span_end_exclusive - span_start
    if len(tokens) > duration:
        raise TextAlignmentV2Error(
            "assistant audio span is too short for one speech-anchor token per frame"
        )
    positions = tuple(
        span_start + (index * duration) // len(tokens)
        for index in range(len(tokens))
    )
    ids = [padding_id] * frame_count
    for frame, token in zip(positions, tokens, strict=True):
        ids[frame] = token
    return AlignedTextTokensV2(
        ids=tuple(ids),
        semantic_tokens=tokens,
        token_frames=positions,
        span_start=span_start,
        span_end_exclusive=span_end_exclusive,
        padding_id=padding_id,
        text_cardinality=text_cardinality,
    )


@dataclass(frozen=True, slots=True)
class TextTokenSpanV2:
    token_id: int
    begin: int
    end: int


@dataclass(frozen=True, slots=True)
class WordStartV2:
    begin: int
    end: int
    start_frame: int


def align_text_tokens_to_sapi_words(
    frame_count: int,
    transcript: str,
    token_spans: Iterable[TextTokenSpanV2],
    word_starts: Iterable[WordStartV2],
    *,
    span_start: int,
    span_end_exclusive: int,
    padding_id: int = 3,
    text_cardinality: int = 32_000,
) -> AlignedTextTokensV2:
    """Align SentencePiece spans to deterministic SAPI word-start timing.

    Pieces owned by one word are spread between that word and the next word's
    start. Punctuation attaches to the previous word; standalone whitespace
    attaches to the next. A final monotonic projection guarantees one token per
    80 ms frame while preserving the measured timing shape.
    """

    if not isinstance(transcript, str) or not transcript:
        raise TextAlignmentV2Error("SAPI alignment requires a non-empty transcript")
    tokens = tuple(token_spans)
    words = tuple(word_starts)
    if not tokens or not words:
        raise TextAlignmentV2Error("SAPI alignment requires token and word spans")
    if span_end_exclusive - span_start < len(tokens):
        raise TextAlignmentV2Error(
            "assistant audio span is too short for one speech-anchor token per frame"
        )
    for token in tokens:
        if (
            isinstance(token.token_id, bool)
            or token.token_id < 0
            or token.token_id >= text_cardinality
            or token.begin < 0
            or token.end <= token.begin
            or token.end > len(transcript)
        ):
            raise TextAlignmentV2Error("invalid SentencePiece token span")
    if any(
        words[index].begin < 0
        or words[index].end <= words[index].begin
        or words[index].end > len(transcript)
        or words[index].start_frame < span_start
        or words[index].start_frame >= span_end_exclusive
        or (
            index > 0
            and (
                words[index].begin < words[index - 1].end
                or words[index].start_frame <= words[index - 1].start_frame
            )
        )
        for index in range(len(words))
    ):
        raise TextAlignmentV2Error("invalid or unordered SAPI word starts")

    owned: list[list[int]] = [[] for _ in words]
    for token_index, token in enumerate(tokens):
        overlaps = [
            (
                max(0, min(token.end, word.end) - max(token.begin, word.begin)),
                word_index,
            )
            for word_index, word in enumerate(words)
        ]
        overlap, owner = max(overlaps, key=lambda item: (item[0], -item[1]))
        if overlap == 0:
            previous = [
                index for index, word in enumerate(words) if word.end <= token.begin
            ]
            following = [
                index for index, word in enumerate(words) if word.begin >= token.end
            ]
            surface = transcript[token.begin : token.end]
            if surface.isspace() and following:
                owner = following[0]
            elif previous:
                owner = previous[-1]
            elif following:
                owner = following[0]
            else:
                raise TextAlignmentV2Error("token span cannot be bound to a SAPI word")
        owned[owner].append(token_index)

    desired = [span_start] * len(tokens)
    for word_index, token_indices in enumerate(owned):
        if not token_indices:
            continue
        start = words[word_index].start_frame
        end = (
            words[word_index + 1].start_frame
            if word_index + 1 < len(words)
            else span_end_exclusive
        )
        duration = max(1, end - start)
        for within_word, token_index in enumerate(token_indices):
            desired[token_index] = start + (
                within_word * duration
            ) // len(token_indices)

    positions: list[int] = []
    for index, target in enumerate(desired):
        lower = span_start if not positions else positions[-1] + 1
        upper = span_end_exclusive - (len(tokens) - index)
        if lower > upper:
            raise TextAlignmentV2Error("SAPI timing cannot fit unique token frames")
        positions.append(min(max(target, lower), upper))
    ids = [padding_id] * frame_count
    for frame, token in zip(positions, tokens, strict=True):
        ids[frame] = token.token_id
    return AlignedTextTokensV2(
        ids=tuple(ids),
        semantic_tokens=tuple(token.token_id for token in tokens),
        token_frames=tuple(positions),
        span_start=span_start,
        span_end_exclusive=span_end_exclusive,
        padding_id=padding_id,
        text_cardinality=text_cardinality,
        method=SAPI_WORD_ALIGNMENT_METHOD,
    )


__all__ = [
    "AlignedTextTokensV2",
    "SAPI_WORD_ALIGNMENT_METHOD",
    "SUPPORTED_TEXT_ALIGNMENT_METHODS",
    "TEXT_ALIGNMENT_METHOD",
    "TextTokenSpanV2",
    "TextAlignmentV2Error",
    "WordStartV2",
    "align_text_tokens_to_sapi_words",
    "uniformly_align_text_tokens",
]
