"""Deterministic supervision masks for imbalanced V2 action timelines."""

from __future__ import annotations

from dataclasses import dataclass

from .action_v2 import DEFAULT_MANIFEST, MicroLane, PAD_TOKEN_ID


class TrainingMaskV2Error(ValueError):
    """Raised when an action-supervision mask cannot be derived safely."""


@dataclass(frozen=True, slots=True)
class ActionSupervisionMaskV2:
    mask: tuple[tuple[bool, ...], ...]
    semantic_token_count: int
    noop_token_count: int
    non_noop_token_count: int
    strategy: str = "all_calls_periodic_noop_v1"

    def __post_init__(self) -> None:
        mask = tuple(tuple(frame) for frame in self.mask)
        if not mask or any(len(frame) != 5 for frame in mask):
            raise TrainingMaskV2Error("action supervision mask must be non-empty [T,5]")
        if any(not isinstance(value, bool) for frame in mask for value in frame):
            raise TrainingMaskV2Error("action supervision mask values must be booleans")
        selected = sum(sum(frame) for frame in mask)
        if selected != self.semantic_token_count:
            raise TrainingMaskV2Error("action supervision token counts do not match the mask")
        if self.noop_token_count + self.non_noop_token_count != selected:
            raise TrainingMaskV2Error("NOOP/non-NOOP counts do not cover selected tokens")
        if self.strategy != "all_calls_periodic_noop_v1":
            raise TrainingMaskV2Error("unknown action supervision strategy")
        object.__setattr__(self, "mask", mask)

    @property
    def frame_count(self) -> int:
        return len(self.mask)

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "mask": [list(frame) for frame in self.mask],
            "semantic_token_count": self.semantic_token_count,
            "noop_token_count": self.noop_token_count,
            "non_noop_token_count": self.non_noop_token_count,
        }


def build_balanced_action_supervision(
    lane: MicroLane,
    *,
    noop_stride_frames: int = 4,
    boundary_noop_frames: int = 2,
) -> ActionSupervisionMaskV2:
    """Score every call token plus a deterministic subset of idle NOOP frames.

    Tool packets are rare compared with 80 ms idle targets.  Scoring every
    NOOP in long result-grounded episodes can teach the fresh action head to
    suppress calls before it learns tool/argument semantics.  This mask keeps
    all non-NOOP tokens, periodic idle evidence, timeline boundaries, and a
    small NOOP window immediately around each action packet.
    """

    if not isinstance(lane, MicroLane):
        raise TrainingMaskV2Error("balanced action supervision requires a MicroLane")
    if (
        isinstance(noop_stride_frames, bool)
        or not isinstance(noop_stride_frames, int)
        or noop_stride_frames < 1
    ):
        raise TrainingMaskV2Error("noop_stride_frames must be a positive integer")
    if (
        isinstance(boundary_noop_frames, bool)
        or not isinstance(boundary_noop_frames, int)
        or boundary_noop_frames < 0
    ):
        raise TrainingMaskV2Error("boundary_noop_frames must be non-negative")

    noop_id = DEFAULT_MANIFEST.action_id("NOOP")
    action_frames = {
        frame_index
        for frame_index, (ids, present) in enumerate(zip(lane.ids, lane.mask, strict=True))
        if any(
            scored and token not in (PAD_TOKEN_ID, noop_id)
            for token, scored in zip(ids, present, strict=True)
        )
    }
    keep_noop_frames = {
        frame
        for frame in range(lane.frame_count)
        if frame % noop_stride_frames == 0
    }
    keep_noop_frames.update({0, lane.frame_count - 1})
    for action_frame in action_frames:
        for offset in range(-boundary_noop_frames, boundary_noop_frames + 1):
            candidate = action_frame + offset
            if 0 <= candidate < lane.frame_count:
                keep_noop_frames.add(candidate)

    result: list[tuple[bool, ...]] = []
    noop_count = 0
    non_noop_count = 0
    for frame_index, (ids, present) in enumerate(zip(lane.ids, lane.mask, strict=True)):
        frame_mask: list[bool] = []
        for token, semantic in zip(ids, present, strict=True):
            if not semantic:
                frame_mask.append(False)
            elif token != noop_id:
                frame_mask.append(True)
                non_noop_count += 1
            elif frame_index in keep_noop_frames:
                frame_mask.append(True)
                noop_count += 1
            else:
                frame_mask.append(False)
        result.append(tuple(frame_mask))

    selected = noop_count + non_noop_count
    if selected < 1:
        raise TrainingMaskV2Error("balanced action supervision selected no targets")
    return ActionSupervisionMaskV2(
        mask=tuple(result),
        semantic_token_count=selected,
        noop_token_count=noop_count,
        non_noop_token_count=non_noop_count,
    )


__all__ = [
    "ActionSupervisionMaskV2",
    "TrainingMaskV2Error",
    "build_balanced_action_supervision",
]
