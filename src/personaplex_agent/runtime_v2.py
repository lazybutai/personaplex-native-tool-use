"""Runtime grammar mask for the V2 five-slot action micro-decoder.

This is deterministic validation/state tracking, not a planner.  It only
removes grammar-invalid logits and commits complete packets after the model has
sampled them.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import torch

from .action_v2 import (
    ActionGrammarError,
    ActionPacketV2,
    ActionParser,
    ActionV2Manifest,
    DEFAULT_MANIFEST,
    MICRO_TOKENS_PER_FRAME,
    PAD_TOKEN_ID,
    ParsedAction,
)


class ActionV2RuntimeConstraint:
    """Stateful batch grammar mask spanning the even/odd packet frames."""

    def __init__(
        self,
        batch_size: int,
        *,
        manifest: ActionV2Manifest = DEFAULT_MANIFEST,
        active_refs: Iterable[Iterable[int]] | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.manifest = manifest
        self.parser = ActionParser(manifest)
        self.batch_size = batch_size
        if active_refs is None:
            self._active_refs = [set() for _ in range(batch_size)]
        else:
            refs = [set(values) for values in active_refs]
            if len(refs) != batch_size:
                raise ValueError("active_refs must have one set per batch item")
            self._active_refs = refs
        self._open_prefix: list[tuple[int, ...] | None] = [None] * batch_size
        self._packet_start: list[int | None] = [None] * batch_size
        self._completed: deque[tuple[int, ParsedAction]] = deque()

        self._id = manifest.action_id
        self._start_tokens = {
            self._id("NOOP"),
            self._id("CALL_BEGIN"),
            self._id("BACKCHANNEL"),
            self._id("PAUSE"),
            self._id("INTERRUPT"),
            self._id("REQUEST_CONFIRM"),
        }

    def set_active_refs(self, batch_index: int, refs: Iterable[int]) -> None:
        self._check_batch_index(batch_index)
        normalized = set(refs)
        if any(ref < 0 or ref >= self.manifest.max_refs for ref in normalized):
            raise ValueError("active REF is outside the V2 range")
        self._active_refs[batch_index] = normalized

    def _check_batch_index(self, batch_index: int) -> None:
        if batch_index < 0 or batch_index >= self.batch_size:
            raise ValueError("batch index is outside the grammar state")

    def _semantic_prefix(self, tokens: torch.Tensor, slot: int) -> tuple[int, ...]:
        prefix = [int(value) for value in tokens[:slot].tolist()]
        if PAD_TOKEN_ID in prefix:
            prefix = prefix[: prefix.index(PAD_TOKEN_ID)]
        return tuple(prefix)

    def allowed_token_ids(
        self,
        batch_index: int,
        phase: int,
        current_frame_prefix: Iterable[int] = (),
    ) -> frozenset[int]:
        """Return exact next-token IDs for one batch item and packet phase."""

        self._check_batch_index(batch_index)
        if phase not in (0, 1):
            raise ValueError("packet phase must be 0 (even) or 1 (odd)")
        current = tuple(current_frame_prefix)
        if any(token < 0 or token >= 256 for token in current):
            raise ValueError("runtime grammar prefix must contain semantic V2 IDs")
        open_prefix = self._open_prefix[batch_index]
        if phase == 0:
            if open_prefix is not None:
                raise ActionGrammarError("an unfinished even-frame call requires its odd continuation")
            prefix = current
        elif open_prefix is None:
            # The logical packet already completed on its even frame.  Its odd
            # physical frame carries only the normal idle target.
            return frozenset({self._id("NOOP")})
        else:
            prefix = open_prefix + current

        if not prefix:
            allowed = set(self._start_tokens)
            if self._active_refs[batch_index]:
                allowed.add(self._id("CANCEL"))
            return frozenset(allowed)

        names = tuple(self.manifest.action_name(token) for token in prefix)
        first = names[0]
        if first in {
            "NOOP",
            "BACKCHANNEL",
            "PAUSE",
            "INTERRUPT",
            "REQUEST_CONFIRM",
        }:
            return frozenset({self._id("NOOP")})

        if first == "CANCEL":
            if len(names) == 1:
                return frozenset(
                    self._id(f"REF_{ref}") for ref in self._active_refs[batch_index]
                )
            if len(names) == 2 and names[1].startswith("REF_"):
                return frozenset({self._id("ACTION_END")})
            return frozenset({self._id("NOOP")})

        if first != "CALL_BEGIN":
            raise ActionGrammarError("invalid V2 action prefix")
        if len(names) == 1:
            return frozenset(
                self.manifest.action_id(tool.action_token)
                for tool in self.manifest.tool_schemas
            )
        try:
            tool = self.manifest.tool_for_action_token(names[1])
        except ActionGrammarError:
            raise ActionGrammarError("CALL_BEGIN must be followed by a tool token")
        if len(names) == 2:
            return frozenset(
                self._id(f"REF_{ref}")
                for ref in range(self.manifest.max_refs)
                if ref not in self._active_refs[batch_index]
            )
        if not names[2].startswith("REF_"):
            raise ActionGrammarError("tool token must be followed by a REF token")
        argument_index = len(names) - 3
        if argument_index < len(tool.arguments):
            return frozenset(
                self.manifest.action_id(token)
                for token in tool.arguments[argument_index].allowed_tokens
            )
        if argument_index == len(tool.arguments):
            return frozenset({self._id("CALL_END")})
        return frozenset({self._id("NOOP")})

    def mask_logits(
        self,
        slot: int,
        phase: torch.Tensor,
        current_tokens: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Callback compatible with ``AgentMicroLMGen.action_logit_masker``."""

        if current_tokens.shape != (self.batch_size, MICRO_TOKENS_PER_FRAME):
            raise ValueError("current action frame shape does not match grammar batch")
        if phase.shape != (self.batch_size,) or logits.shape != (
            self.batch_size,
            self.manifest.action_cardinality,
        ):
            raise ValueError("grammar callback received incompatible phase/logit shapes")
        masked = torch.full_like(logits, float("-inf"))
        for batch_index in range(self.batch_size):
            allowed = self.allowed_token_ids(
                batch_index,
                int(phase[batch_index].item()),
                self._semantic_prefix(current_tokens[batch_index], slot),
            )
            if not allowed:
                raise ActionGrammarError("V2 grammar has no legal next token")
            indices = torch.tensor(
                sorted(allowed), dtype=torch.long, device=logits.device
            )
            masked[batch_index, indices] = logits[batch_index, indices]
        return masked

    def commit_frame(
        self,
        frame_index: torch.Tensor,
        phase: torch.Tensor,
        action_tokens: torch.Tensor,
    ) -> None:
        """Observer compatible with ``AgentMicroLMGen.action_frame_observer``."""

        expected = (self.batch_size, MICRO_TOKENS_PER_FRAME)
        if action_tokens.shape != expected:
            raise ValueError("committed V2 action frame has the wrong shape")
        if frame_index.shape != (self.batch_size,) or phase.shape != (self.batch_size,):
            raise ValueError("committed V2 frame metadata has the wrong shape")
        for batch_index in range(self.batch_size):
            frame = int(frame_index[batch_index].item())
            frame_phase = int(phase[batch_index].item())
            if frame % 2 != frame_phase:
                raise ActionGrammarError("runtime frame index disagrees with packet phase")
            values = [int(token) for token in action_tokens[batch_index].tolist()]
            semantic = tuple(token for token in values if token != PAD_TOKEN_ID)
            if frame_phase == 0:
                packet_start = frame
                if semantic and semantic[0] == self._id("CALL_BEGIN") and (
                    self._id("CALL_END") not in semantic
                ):
                    self._open_prefix[batch_index] = semantic
                    self._packet_start[batch_index] = packet_start
                    continue
                parsed = self.parser.parse(
                    ActionPacketV2(packet_start, semantic),
                    active_refs=self._active_refs[batch_index],
                )
                self._completed.append((batch_index, parsed))
            else:
                open_prefix = self._open_prefix[batch_index]
                if open_prefix is None:
                    if semantic != (self._id("NOOP"),):
                        raise ActionGrammarError(
                            "odd frame without an open call must contain only NOOP"
                        )
                    continue
                packet_start = self._packet_start[batch_index]
                assert packet_start is not None
                parsed = self.parser.parse(
                    ActionPacketV2(packet_start, open_prefix + semantic),
                    active_refs=self._active_refs[batch_index],
                )
                self._completed.append((batch_index, parsed))
                self._open_prefix[batch_index] = None
                self._packet_start[batch_index] = None

    def drain_completed(self) -> tuple[tuple[int, ParsedAction], ...]:
        completed = tuple(self._completed)
        self._completed.clear()
        return completed


__all__ = ["ActionV2RuntimeConstraint"]
