"""Strict frame interleaving for native action-lane training.

The v1 compatibility path has one generated action token per 80 ms frame.  The
micro-decoder path packs up to five generated tokens into each 80 ms frame,
matching the ten-token/160 ms contract.  Environment packets remain trusted
input-only events serialized one token per frame with collision checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping

from .dataset import DuplexToolEpisode
from .native_lane import MAX_ACTION_TOKENS_PER_FRAME


NO_ACTION_TOKEN_ID = 0
ACTION_PADDING_TOKEN_ID = -1
ENVIRONMENT_ABSENT_TOKEN_ID = -1


class ActionMicrodecoderRequiredError(ValueError):
    """Raised when a rich action packet cannot fit the current v1 action head."""


@dataclass(frozen=True, slots=True)
class TypedLaneVocabulary:
    """Frozen symbolic-token vocabulary for the v1 one-token-per-frame lanes.

    Action ID 0 is a supervised no-op.  Environment ID -1 is an absent
    executor observation and maps to the model's zero embedding; environment
    tokens begin at 0 because they are input-only rather than model targets.
    """

    version: str
    action_tokens: Mapping[str, int]
    environment_tokens: Mapping[str, int]

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("vocabulary version is required")
        self._validate_mapping(self.action_tokens, start=1, lane="action")
        self._validate_mapping(self.environment_tokens, start=0, lane="environment")

    @staticmethod
    def _validate_mapping(mapping: Mapping[str, int], *, start: int, lane: str) -> None:
        if not mapping:
            raise ValueError(f"{lane} vocabulary cannot be empty")
        if any(not token for token in mapping):
            raise ValueError(f"{lane} vocabulary contains an empty token")
        values = sorted(mapping.values())
        expected = list(range(start, start + len(values)))
        if values != expected:
            raise ValueError(
                f"{lane} IDs must be a contiguous frozen range {expected}, got {values}"
            )

    @classmethod
    def build(
        cls,
        *,
        version: str,
        action_tokens: Iterable[str],
        environment_tokens: Iterable[str],
    ) -> "TypedLaneVocabulary":
        unique_actions = sorted(set(action_tokens))
        unique_environment = sorted(set(environment_tokens))
        return cls(
            version=version,
            action_tokens={token: index + 1 for index, token in enumerate(unique_actions)},
            environment_tokens={token: index for index, token in enumerate(unique_environment)},
        )

    @property
    def action_card(self) -> int:
        """Cardinality passed to ``AgentLMModel(action_card=...)``."""

        return len(self.action_tokens) + 1

    @property
    def environment_card(self) -> int:
        """Cardinality passed to ``AgentLMModel(environment_card=...)``."""

        # The model reserves one extra in-vocabulary slot for padding/context
        # and enforces a minimum cardinality of two.
        return max(2, len(self.environment_tokens))

    @property
    def fingerprint(self) -> str:
        serialized = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def action_id(self, token: str) -> int:
        try:
            return self.action_tokens[token]
        except KeyError as exc:
            raise ValueError(f"unknown frozen action token: {token!r}") from exc

    def environment_id(self, token: str) -> int:
        try:
            return self.environment_tokens[token]
        except KeyError as exc:
            raise ValueError(f"unknown frozen environment token: {token!r}") from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "action_tokens": dict(self.action_tokens),
            "environment_tokens": dict(self.environment_tokens),
        }


@dataclass(frozen=True, slots=True)
class V1FrameTargets:
    """Raw, pre-delay target/input frames for ``AgentLMModel.forward_agent_train``."""

    action_tokens: tuple[int, ...]
    environment_tokens: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.action_tokens:
            raise ValueError("at least one 80 ms frame is required")
        if len(self.action_tokens) != len(self.environment_tokens):
            raise ValueError("action and environment timelines must have equal length")
        if any(token < NO_ACTION_TOKEN_ID for token in self.action_tokens):
            raise ValueError("action targets cannot use a negative sentinel")
        if any(token < ENVIRONMENT_ABSENT_TOKEN_ID for token in self.environment_tokens):
            raise ValueError("environment target is outside the typed-lane domain")

    @property
    def frame_count(self) -> int:
        return len(self.action_tokens)


@dataclass(frozen=True, slots=True)
class MicroFrameTargets:
    """Raw frame targets for the bounded action micro-decoder.

    Padding is a negative cache/trailing filler and is never a model target.
    The structured grammar supplies explicit terminal tokens such as
    ``CALL_END``/``ACTION_END``; padding is never exposed to the executor.
    """

    action_tokens: tuple[tuple[int, ...], ...]
    environment_tokens: tuple[int, ...]
    action_padding_id: int

    def __post_init__(self) -> None:
        if not self.action_tokens:
            raise ValueError("at least one 80 ms frame is required")
        if len(self.action_tokens) != len(self.environment_tokens):
            raise ValueError("action and environment timelines must have equal length")
        if self.action_padding_id != ACTION_PADDING_TOKEN_ID:
            raise ValueError("micro-decoder action padding must use the fixed -1 sentinel")
        for frame in self.action_tokens:
            if len(frame) != MAX_ACTION_TOKENS_PER_FRAME:
                raise ValueError("every micro-decoder frame must have five action slots")
            padding_started = False
            for token in frame:
                if token == self.action_padding_id:
                    padding_started = True
                elif padding_started:
                    raise ValueError("action padding must be a contiguous frame suffix")
                elif token < NO_ACTION_TOKEN_ID:
                    raise ValueError("action micro-token is outside its vocabulary")
        if any(token < ENVIRONMENT_ABSENT_TOKEN_ID for token in self.environment_tokens):
            raise ValueError("environment target is outside the typed-lane domain")

    @property
    def frame_count(self) -> int:
        return len(self.action_tokens)

    @property
    def action_mask(self) -> tuple[tuple[bool, ...], ...]:
        """Score generated semantic tokens only; padding is never predicted."""

        masks: list[tuple[bool, ...]] = []
        for frame in self.action_tokens:
            masks.append(tuple(token != self.action_padding_id for token in frame))
        return tuple(masks)

    def visible_action_tokens(self, frame: int) -> tuple[int, ...]:
        """Return executor-visible tokens with the internal suffix removed."""

        return tuple(token for token in self.action_tokens[frame] if token != self.action_padding_id)


def _build_environment_inputs(
    episode: DuplexToolEpisode,
    vocabulary: TypedLaneVocabulary,
    *,
    frame_count: int,
) -> list[int]:
    environment_inputs = [ENVIRONMENT_ABSENT_TOKEN_ID] * frame_count
    for packet in sorted(episode.environment_packets, key=lambda item: (item.frame, item.call_id)):
        for offset, token in enumerate(packet.tokens):
            frame = packet.frame + offset
            if frame >= frame_count:
                raise ValueError("environment packet exceeds the available aligned audio frames")
            if environment_inputs[frame] != ENVIRONMENT_ABSENT_TOKEN_ID:
                raise ValueError(f"two executor observations collide at frame {frame}")
            environment_inputs[frame] = vocabulary.environment_id(token)
    return environment_inputs


def build_v1_frame_targets(
    episode: DuplexToolEpisode,
    vocabulary: TypedLaneVocabulary,
    *,
    frame_count: int,
) -> V1FrameTargets:
    """Encode an episode without losing action/result timing information.

    The returned lanes are *raw* 80 ms frames.  ``AgentLMModel`` is responsible
    for applying the one-step causal delay before any model forward pass.
    """

    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    episode.validate()
    action_targets = [NO_ACTION_TOKEN_ID] * frame_count
    environment_inputs = _build_environment_inputs(
        episode, vocabulary, frame_count=frame_count
    )

    for packet in episode.action_packets:
        for frame in (packet.start_frame, packet.start_frame + 1):
            tokens = packet.tokens_for_frame(frame)
            if len(tokens) > 1:
                raise ActionMicrodecoderRequiredError(
                    "v1 has one generated action token per 80 ms frame; "
                    f"packet at frame {frame} contains {len(tokens)} tokens"
                )
            if not tokens:
                continue
            if frame >= frame_count:
                raise ValueError("action packet exceeds the available aligned audio frames")
            if action_targets[frame] != NO_ACTION_TOKEN_ID:
                raise ValueError(f"two model actions collide at frame {frame}")
            action_targets[frame] = vocabulary.action_id(tokens[0])

    return V1FrameTargets(tuple(action_targets), tuple(environment_inputs))


def build_micro_frame_targets(
    episode: DuplexToolEpisode,
    vocabulary: TypedLaneVocabulary,
    *,
    frame_count: int,
) -> MicroFrameTargets:
    """Encode the full five-token-per-frame structured action contract.

    Each frame defaults to ``NO_ACTION`` followed by an internal stop/padding
    suffix.  Rich packets are split by :class:`ActionPacket` across their two
    80 ms frames without truncation.  These are raw logical frames; the model
    applies the temporal one-step delay exactly once.
    """

    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    episode.validate()
    padding_id = ACTION_PADDING_TOKEN_ID
    empty_frame = [NO_ACTION_TOKEN_ID] + [padding_id] * (
        MAX_ACTION_TOKENS_PER_FRAME - 1
    )
    action_targets = [list(empty_frame) for _ in range(frame_count)]
    occupied_frames: set[int] = set()

    for packet in episode.action_packets:
        for frame in (packet.start_frame, packet.start_frame + 1):
            tokens = packet.tokens_for_frame(frame)
            if not tokens:
                continue
            if frame >= frame_count:
                raise ValueError("action packet exceeds the available aligned audio frames")
            if frame in occupied_frames:
                raise ValueError(f"two model action packets collide at frame {frame}")
            encoded = [vocabulary.action_id(token) for token in tokens]
            if len(encoded) > MAX_ACTION_TOKENS_PER_FRAME:
                raise ValueError("action frame exceeds the micro-decoder slot budget")
            action_targets[frame] = encoded + [padding_id] * (
                MAX_ACTION_TOKENS_PER_FRAME - len(encoded)
            )
            occupied_frames.add(frame)

    environment_inputs = _build_environment_inputs(
        episode, vocabulary, frame_count=frame_count
    )
    return MicroFrameTargets(
        action_tokens=tuple(tuple(frame) for frame in action_targets),
        environment_tokens=tuple(environment_inputs),
        action_padding_id=padding_id,
    )
