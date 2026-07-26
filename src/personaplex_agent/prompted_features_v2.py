"""Prompt-conditioned action-feature timeline contracts.

The 7B PersonaPlex backbone is frozen during the first useful action stage.  We
therefore cache its real prompted streaming states and train only the native V2
action micro-decoder on those states.  This module builds the causal timelines
used by that extractor without importing Torch or the model fork.

Timing augmentation is intentional: user speech may begin at any 80 ms frame,
while V2 calls still begin on the next legal even packet frame.  Inserted
silence and the post-result tail receive explicit NOOP supervision so absolute
frame position and repeated calls cannot solve the task.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .action_v2 import DEFAULT_MANIFEST, MICRO_TOKENS_PER_FRAME, PAD_TOKEN_ID


FEATURE_VERSION = "ppx-prompted-action-feature-v1.0.0"


class PromptedFeatureV2Error(ValueError):
    """Raised before an invalid augmented timeline reaches GPU extraction."""


@dataclass(frozen=True, slots=True)
class PromptedTimelineV2:
    user_codes: np.ndarray  # [8,T]
    action_ids: np.ndarray  # [T,5]
    action_mask: np.ndarray  # [T,5]
    action_supervision_mask: np.ndarray  # [T,5]
    environment_ids: np.ndarray  # [T,5]
    environment_mask: np.ndarray  # [T,5]
    user_start_frame: int
    user_end_frame: int
    call_start_frame: int | None
    source_call_start_frame: int | None
    lead_frames: int
    tail_frames: int

    def __post_init__(self) -> None:
        user_codes = np.asarray(self.user_codes)
        actions = np.asarray(self.action_ids)
        action_mask = np.asarray(self.action_mask)
        supervision = np.asarray(self.action_supervision_mask)
        environment = np.asarray(self.environment_ids)
        environment_mask = np.asarray(self.environment_mask)
        if user_codes.ndim != 2 or user_codes.shape[0] != 8:
            raise PromptedFeatureV2Error("prompted user codes must be [8,T]")
        frames = user_codes.shape[1]
        expected = (frames, MICRO_TOKENS_PER_FRAME)
        if actions.shape != expected or environment.shape != expected:
            raise PromptedFeatureV2Error("typed prompted lanes must be [T,5]")
        if action_mask.shape != expected or supervision.shape != expected:
            raise PromptedFeatureV2Error("prompted action masks must be [T,5]")
        if environment_mask.shape != expected:
            raise PromptedFeatureV2Error("prompted environment mask must be [T,5]")
        if not np.array_equal(action_mask, actions >= 0):
            raise PromptedFeatureV2Error("action mask must exactly select semantic IDs")
        if not np.array_equal(environment_mask, environment >= 0):
            raise PromptedFeatureV2Error("environment mask must exactly select semantic IDs")
        if np.any(supervision & ~action_mask):
            raise PromptedFeatureV2Error("action supervision cannot select PAD")
        if not np.any(supervision):
            raise PromptedFeatureV2Error("prompted timeline has no action supervision")
        if not 0 <= self.user_start_frame <= self.user_end_frame <= frames:
            raise PromptedFeatureV2Error("prompted user span is outside the timeline")
        if self.call_start_frame is not None:
            if self.call_start_frame < self.user_end_frame:
                raise PromptedFeatureV2Error("tool call begins before user audio ends")
            if self.call_start_frame % 2:
                raise PromptedFeatureV2Error("V2 tool call must begin on an even frame")
        for name in ("lead_frames", "tail_frames"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PromptedFeatureV2Error(f"{name} must be a non-negative integer")

        # Freeze caller-owned arrays against accidental in-memory drift.
        for name, value in (
            ("user_codes", user_codes),
            ("action_ids", actions),
            ("action_mask", action_mask),
            ("action_supervision_mask", supervision),
            ("environment_ids", environment),
            ("environment_mask", environment_mask),
        ):
            copied = np.ascontiguousarray(value).copy()
            copied.flags.writeable = False
            object.__setattr__(self, name, copied)

    @property
    def frame_count(self) -> int:
        return int(self.user_codes.shape[1])

    def arrays(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType(
            {
                "user_codes": self.user_codes,
                "action_ids": self.action_ids,
                "action_mask": self.action_mask,
                "action_supervision_mask": self.action_supervision_mask,
                "environment_ids": self.environment_ids,
                "environment_mask": self.environment_mask,
            }
        )


def next_even_frame(frame: int) -> int:
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise PromptedFeatureV2Error("frame must be a non-negative integer")
    return frame if frame % 2 == 0 else frame + 1


def build_prompted_timeline(
    user_codes: np.ndarray,
    *,
    source_action_ids: np.ndarray,
    source_action_supervision_mask: np.ndarray,
    source_environment_ids: np.ndarray,
    source_call_start_frame: int | None,
    lead_frames: int,
    tail_frames: int,
    silence_tokens: np.ndarray,
) -> PromptedTimelineV2:
    """Move one labeled scenario onto a randomized live-audio timeline."""

    user = np.asarray(user_codes)
    source_actions = np.asarray(source_action_ids)
    source_supervision = np.asarray(source_action_supervision_mask)
    source_environment = np.asarray(source_environment_ids)
    silence = np.asarray(silence_tokens)
    if user.ndim != 2 or user.shape[0] != 8 or user.shape[1] < 1:
        raise PromptedFeatureV2Error("source user codes must be non-empty [8,T]")
    if source_actions.ndim != 2 or source_actions.shape[1] != MICRO_TOKENS_PER_FRAME:
        raise PromptedFeatureV2Error("source action lane must be [T,5]")
    if source_supervision.shape != source_actions.shape:
        raise PromptedFeatureV2Error("source action supervision must align with actions")
    if source_environment.shape != source_actions.shape:
        raise PromptedFeatureV2Error("source environment lane must align with actions")
    if silence.shape != (8,):
        raise PromptedFeatureV2Error("silence tokens must contain eight Mimi codebooks")
    if lead_frames < 0 or tail_frames < 1:
        raise PromptedFeatureV2Error("lead must be non-negative and tail must be positive")

    noop_id = DEFAULT_MANIFEST.action_id("NOOP")
    user_start = lead_frames
    user_end = user_start + int(user.shape[1])
    call_start: int | None
    if source_call_start_frame is None:
        suffix_length = 0
        call_start = None
        frames = user_end + tail_frames
    else:
        if (
            source_call_start_frame < 0
            or source_call_start_frame >= source_actions.shape[0]
            or source_call_start_frame % 2
        ):
            raise PromptedFeatureV2Error("source call frame must be a valid even frame")
        suffix_length = source_actions.shape[0] - source_call_start_frame
        call_start = next_even_frame(user_end)
        frames = call_start + suffix_length + tail_frames

    timeline_user = np.repeat(silence[:, None], frames, axis=1).astype(
        user.dtype, copy=False
    )
    timeline_user[:, user_start:user_end] = user
    actions = np.full(
        (frames, MICRO_TOKENS_PER_FRAME), PAD_TOKEN_ID, dtype=source_actions.dtype
    )
    actions[:, 0] = noop_id
    environment = np.full(
        (frames, MICRO_TOKENS_PER_FRAME),
        PAD_TOKEN_ID,
        dtype=source_environment.dtype,
    )
    supervision = np.zeros_like(actions, dtype=np.bool_)

    # CALL_BEGIN is legal only on even frames. Dense even-frame NOOP targets
    # directly penalize pre-speech hallucinations and post-result repetitions.
    supervision[0::2, 0] = True
    supervision[-1, 0] = True
    if call_start is not None and source_call_start_frame is not None:
        source_slice = slice(source_call_start_frame, None)
        target_slice = slice(call_start, call_start + suffix_length)
        actions[target_slice] = source_actions[source_slice]
        environment[target_slice] = source_environment[source_slice]
        supervision[target_slice] |= source_supervision[source_slice]
        supervision[target_slice] |= actions[target_slice] > noop_id

    return PromptedTimelineV2(
        user_codes=timeline_user,
        action_ids=actions,
        action_mask=actions >= 0,
        action_supervision_mask=supervision,
        environment_ids=environment,
        environment_mask=environment >= 0,
        user_start_frame=user_start,
        user_end_frame=user_end,
        call_start_frame=call_start,
        source_call_start_frame=source_call_start_frame,
        lead_frames=lead_frames,
        tail_frames=tail_frames,
    )


def feature_fingerprint(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]
) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    stable_metadata = {
        key: metadata[key]
        for key in sorted(metadata)
        if key != "feature_fingerprint"
    }
    digest.update(
        json.dumps(stable_metadata, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    return digest.hexdigest()


__all__ = [
    "FEATURE_VERSION",
    "PromptedFeatureV2Error",
    "PromptedTimelineV2",
    "build_prompted_timeline",
    "feature_fingerprint",
    "next_even_frame",
]
