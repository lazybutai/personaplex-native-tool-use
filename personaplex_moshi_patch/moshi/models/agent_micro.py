"""Five-slot native action micro-decoder for PersonaPlex-Agent.

The temporal backbone advances once per 80 ms PersonaPlex frame.  It consumes
the previous frame's ordered action/environment microtokens, then predicts:

1. assistant text;
2. up to five autoregressive action tokens;
3. audio depth tokens conditioned on the same-tick sampled action summary.

Environment tokens are executor-authored inputs only.  This module has no
environment output head and no environment loss path.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional

import torch

from ..modules.streaming import StreamingModule
from ..modules.transformer import StreamingTransformer
from ..utils.compile import CUDAGraphed, in_cuda_graph
from ..utils.sampling import sample_token
from .agent_lm import AgentLMModel, AgentLossReport, _masked_cross_entropy
from .lm import _delay_sequence, _undelay_sequence


ACTION_SLOTS_PER_FRAME = 5
ENVIRONMENT_SLOTS_PER_FRAME = 5
MICRO_PADDING_TOKEN_ID = -1

ActionLogitMasker = Callable[
    [int, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]
ActionFrameObserver = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]
BackboneFrameObserver = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]
PendingSpeechObserver = Callable[[torch.Tensor, torch.Tensor], None]
TurnCompletionObserver = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor], None
]
TextSamplingObserver = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    None,
]
ResultOnsetSelectionObserver = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    None,
]


@dataclass(frozen=True)
class AgentMicroBackboneOutput:
    transformer_out: torch.Tensor  # action-conditioned state
    base_transformer_out: torch.Tensor  # stock PersonaPlex text/audio state
    speech_transformer_out: torch.Tensor  # optional typed-result-conditioned state
    response_environment_state: torch.Tensor  # cumulative typed-result context
    response_temporal_state: torch.Tensor  # recurrent ordered-response state
    response_terminal_seen: torch.Tensor  # terminal status observed in current event
    response_active: torch.Tensor  # complete terminal event has become visible
    response_token_repair_scope_seen: torch.Tensor  # scoped result marker observed
    response_token_repair_scope_semantic_seen: torch.Tensor  # response onset passed
    text_logits: torch.Tensor
    response_token_repair_bias: Optional[torch.Tensor] = None
    response_token_repair_hidden: Optional[torch.Tensor] = None
    response_token_repair_source: Optional[torch.Tensor] = None
    response_token_continuation_bias: Optional[torch.Tensor] = None
    response_token_continuation_hidden: Optional[torch.Tensor] = None
    response_token_continuation_source: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class ActionMicroOutput:
    tokens: torch.Tensor  # [B, action_slots]
    logits: torch.Tensor  # [B, action_slots, action_card]
    call_gate_logits: torch.Tensor  # [B], endpoint AND semantic intent
    call_endpoint_logits: torch.Tensor  # [B]
    call_intent_logits: torch.Tensor  # [B]
    call_soft_intent_logits: torch.Tensor  # [B]
    audio_summary: torch.Tensor  # [B, depformer_dim]
    temporal_state: torch.Tensor  # [layers,B,action_depth_dim]
    start_temporal_state: Optional[torch.Tensor]  # frozen even-frame start branch
    call_gate_temporal_state: torch.Tensor  # [layers,B,action_depth_dim]
    call_intent_temporal_state: torch.Tensor  # [layers,B,action_depth_dim]


ActionDecisionObserver = Callable[
    [torch.Tensor, torch.Tensor, ActionMicroOutput], None
]


@dataclass(frozen=True)
class AgentMicroTrainOutput:
    audio_logits: torch.Tensor
    audio_mask: torch.Tensor
    text_logits: torch.Tensor
    text_mask: torch.Tensor
    action_logits: torch.Tensor  # [B, T, action_slots, action_card]
    action_mask: torch.Tensor  # [B, T, action_slots]
    response_token_repair_bias: Optional[torch.Tensor] = None
    response_token_repair_hidden: Optional[torch.Tensor] = None
    response_token_repair_source: Optional[torch.Tensor] = None
    response_token_continuation_bias: Optional[torch.Tensor] = None
    response_token_continuation_hidden: Optional[torch.Tensor] = None
    response_token_continuation_source: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class AgentMicroGateTrainOutput:
    transformer_out: torch.Tensor  # [B, T, dim]
    text_logits: torch.Tensor  # [B, 1, T, text_card]
    call_endpoint_logits: torch.Tensor  # [B, T]
    call_intent_logits: torch.Tensor  # [B, T]


class LateBackboneAdapter(torch.nn.Module):
    """Zero-gated action-only residual over the final frozen base state."""

    def __init__(
        self,
        dim: int,
        bottleneck_dim: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if dim < 1 or bottleneck_dim < 1:
            raise ValueError("late-backbone adapter dimensions must be positive")
        factory_kwargs = {"device": device, "dtype": dtype}
        self.norm = torch.nn.LayerNorm(dim, **factory_kwargs)
        self.down = torch.nn.Linear(
            dim, bottleneck_dim, bias=False, **factory_kwargs
        )
        self.up = torch.nn.Linear(
            bottleneck_dim, dim, bias=False, **factory_kwargs
        )
        # The adapter is an exact no-op until training moves this scalar. This
        # protects stock and legacy checkpoints even though the low-rank
        # projections are initialized normally and ready to receive gradients.
        self.gate = torch.nn.Parameter(torch.zeros((), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        self.down.reset_parameters()
        with torch.no_grad():
            # Standard zero-output adapter initialization: the residual begins
            # as an exact no-op while the output projection can learn on the
            # first update and unlock gradients into the down projection.
            self.up.weight.zero_()
            self.gate.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.up(torch.nn.functional.silu(self.down(self.norm(x))))
        return x + torch.tanh(self.gate) * update


class AgentMicroLMModel(AgentLMModel):
    """PersonaPlex plus ordered action and environment microtoken histories."""

    def __init__(
        self,
        *args,
        action_slots: int = ACTION_SLOTS_PER_FRAME,
        environment_slots: int = ENVIRONMENT_SLOTS_PER_FRAME,
        action_end_token_id: int = 4,
        action_call_begin_token_id: int = 1,
        action_call_gate_threshold: float = 0.0,
        action_timer_tool_token_id: int = 14,
        terminal_action_token_ids: tuple[int, ...] = (0, 2, 4, 5, 6, 7, 8),
        action_depth_dim: int = 512,
        action_depth_num_heads: int = 8,
        action_depth_num_layers: int = 2,
        action_depth_hidden_scale: float = 4.0,
        action_backbone_adapter_dim: int = 64,
        response_temporal_dim: int = 256,
        response_token_repair_rank: int = 8,
        response_token_continuation_rank: int = 32,
        response_obs_begin_token_id: int = 0,
        response_obs_end_token_id: int = 1,
        response_terminal_status_token_ids: tuple[int, ...] = (3, 4, 5, 6, 7),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if action_slots != ACTION_SLOTS_PER_FRAME:
            raise ValueError("the v2 contract requires exactly five action slots per 80 ms")
        if environment_slots != ENVIRONMENT_SLOTS_PER_FRAME:
            raise ValueError("the v2 contract requires exactly five environment slots per 80 ms")
        if not 0 < action_end_token_id < self.action_card:
            raise ValueError("ACTION_END must be a generated in-vocabulary token")
        if not 0 < action_call_begin_token_id < self.action_card:
            raise ValueError("CALL_BEGIN must be a generated in-vocabulary token")
        if not math.isfinite(action_call_gate_threshold):
            raise ValueError("action call-gate threshold must be finite")
        if not 0 <= action_timer_tool_token_id < self.action_card:
            raise ValueError("timer tool token is outside the action vocabulary")
        if action_depth_dim % action_depth_num_heads:
            raise ValueError("action depth dimension must be divisible by its head count")
        if action_backbone_adapter_dim < 1:
            raise ValueError("action backbone adapter dimension must be positive")
        if response_temporal_dim < 1:
            raise ValueError("response temporal dimension must be positive")
        if response_token_repair_rank < 1:
            raise ValueError("response token repair rank must be positive")
        if response_token_continuation_rank < 1:
            raise ValueError("response token continuation rank must be positive")
        if not 0 <= response_obs_begin_token_id < self.environment_card:
            raise ValueError("response OBS_BEGIN token is outside the environment vocabulary")
        if not 0 <= response_obs_end_token_id < self.environment_card:
            raise ValueError("response OBS_END token is outside the environment vocabulary")
        if not response_terminal_status_token_ids or any(
            token < 0 or token >= self.environment_card
            for token in response_terminal_status_token_ids
        ):
            raise ValueError("response terminal status is outside the environment vocabulary")

        self.action_slots = action_slots
        self.environment_slots = environment_slots
        self.action_end_token_id = action_end_token_id
        self.action_call_begin_token_id = action_call_begin_token_id
        self.action_call_gate_threshold = float(action_call_gate_threshold)
        self.action_timer_tool_token_id = action_timer_tool_token_id
        if self.no_action_token_id not in terminal_action_token_ids or (
            action_end_token_id not in terminal_action_token_ids
        ):
            raise ValueError("terminal actions must include NO_ACTION and ACTION_END")
        if any(token < 0 or token >= self.action_card for token in terminal_action_token_ids):
            raise ValueError("terminal action token is outside the action vocabulary")
        self.terminal_action_token_ids = frozenset(terminal_action_token_ids)
        self.action_depth_dim = action_depth_dim
        self.response_temporal_dim = response_temporal_dim
        self.response_token_repair_rank = response_token_repair_rank
        self.response_token_continuation_rank = response_token_continuation_rank
        self.response_obs_begin_token_id = response_obs_begin_token_id
        self.response_obs_end_token_id = response_obs_end_token_id
        self.response_terminal_status_token_ids = frozenset(
            response_terminal_status_token_ids
        )
        lane_factory_kwargs = {
            "device": self.text_linear.weight.device,
            "dtype": self.text_linear.weight.dtype,
        }

        # A compact model-owned residual is evaluated on an action-only branch
        # after the frozen PersonaPlex backbone. Stock text/audio consume the
        # untouched base state. Setting ``action_backbone_adapter_detach_input``
        # during training avoids retaining a graph through the frozen backbone.
        self.action_backbone_adapter = LateBackboneAdapter(
            self.dim,
            action_backbone_adapter_dim,
            **lane_factory_kwargs,
        )
        self.action_backbone_adapter_layer = len(self.transformer.layers) - 1
        self.action_backbone_adapter_detach_input = False

        # Slot-specific history embeddings preserve token order when the prior
        # frame is folded into one temporal-backbone input vector.
        self.action_history_emb = torch.nn.ModuleList(
            [self.EmbeddingFactory(self.action_card, self.dim) for _ in range(action_slots)]
        )
        self.environment_history_emb = torch.nn.ModuleList(
            [
                self.EmbeddingFactory(self.environment_card, self.dim)
                for _ in range(environment_slots)
            ]
        )
        # A speech-only copy prevents result-grounding updates from changing
        # the action branch's environment semantics. The zero scale keeps all
        # legacy checkpoints byte-equivalent to stock PersonaPlex.
        self.response_environment_history_emb = torch.nn.ModuleList(
            [
                self.EmbeddingFactory(self.environment_card, self.dim)
                for _ in range(environment_slots)
            ]
        )
        self.response_environment_scale = torch.nn.Parameter(
            torch.zeros((self.dim,), **lane_factory_kwargs)
        )
        self.response_environment_conditioning_enabled = False
        # The speech branch initially sees only typed environment results. This
        # separate exact-zero scale can also fold the model's own non-NOOP call
        # packet into the same causal response context without modifying the
        # action embeddings or the stock PersonaPlex speech state.
        self.response_action_scale = torch.nn.Parameter(
            torch.zeros((self.dim,), **lane_factory_kwargs)
        )
        self.response_action_conditioning_enabled = False
        # A static result bias cannot express an ordered sentence. This compact
        # speech-only recurrent branch evolves with the frozen conversational
        # state and cumulative typed result, while a terminal-completion mask
        # prevents it from affecting speech before OBS_END is visible.
        self.response_temporal_base_in = torch.nn.Linear(
            self.dim, response_temporal_dim, bias=False, **lane_factory_kwargs
        )
        self.response_temporal_context_in = torch.nn.Linear(
            self.dim, response_temporal_dim, bias=False, **lane_factory_kwargs
        )
        self.response_temporal_gru = torch.nn.GRU(
            response_temporal_dim,
            response_temporal_dim,
            num_layers=1,
            batch_first=True,
            **lane_factory_kwargs,
        )
        self.response_temporal_out = torch.nn.Linear(
            response_temporal_dim, self.dim, bias=False, **lane_factory_kwargs
        )
        self.response_temporal_conditioning_enabled = False
        # This zero-disabled scalar head can raise only native text EOS after a
        # terminal result. Stop-timing training cannot rewrite word/audio logits.
        self.response_eos_head = torch.nn.Linear(
            response_temporal_dim, 1, bias=True, **lane_factory_kwargs
        )
        self.response_eos_conditioning_enabled = False
        self.response_eos_token_id = 2
        # A full-vocabulary, model-owned residual can move result words that
        # remain outside native top-k sampling even after hidden-state
        # adaptation. It consumes the same causal temporal result state, is
        # active only after a complete terminal event, and initializes to an
        # exact zero logit delta for migration safety.
        self.response_token_head = torch.nn.Linear(
            response_temporal_dim,
            self.text_linear.weight.shape[0],
            bias=False,
            **lane_factory_kwargs,
        )
        self.response_token_conditioning_enabled = False
        # Optional low-rank access to PersonaPlex's current causal language
        # state lets the result head distinguish adjacent word-piece contexts.
        # It is zero-disabled and affects only the post-result token residual.
        self.response_token_context_in = torch.nn.Linear(
            self.dim,
            response_temporal_dim,
            bias=False,
            **lane_factory_kwargs,
        )
        self.response_token_context_conditioning_enabled = False
        # A tiny state-conditioned repair head avoids moving the global token
        # rows used by unrelated held-out states. Its output projection is
        # exact-zero until explicitly trained, while the fixed-size bottleneck
        # can distinguish different post-result causal histories.
        self.response_token_repair_norm = torch.nn.LayerNorm(
            response_temporal_dim,
            elementwise_affine=False,
            **lane_factory_kwargs,
        )
        self.response_token_repair_down = torch.nn.Linear(
            response_temporal_dim,
            response_token_repair_rank,
            bias=False,
            **lane_factory_kwargs,
        )
        self.response_token_repair_up = torch.nn.Linear(
            response_token_repair_rank,
            self.text_linear.weight.shape[0],
            bias=False,
            **lane_factory_kwargs,
        )
        self.response_token_repair_conditioning_enabled = False
        # A separately migrated affine selector gives the low-rank repair a
        # learnable threshold without changing old repair checkpoints. Zero
        # weight/bias yields 1+0.5*tanh(0)=1, exactly preserving the ungated
        # repair path. The bounded 0.5x-1.5x range prevents the selector from
        # collapsing into a trivial off switch under preservation loss.
        self.response_token_repair_gate = torch.nn.Linear(
            response_temporal_dim,
            1,
            bias=True,
            **lane_factory_kwargs,
        )
        self.response_token_repair_gate_conditioning_enabled = False
        # Optional checkpoint-scoped polarity constraints keep a repair row
        # from suppressing the token it was introduced to repair. Empty by
        # default, so every existing checkpoint remains behavior-identical.
        self.response_token_repair_nonnegative_rows: tuple[int, ...] = ()
        self.register_buffer(
            "response_token_repair_nonnegative_rows_tensor",
            torch.empty(
                0,
                dtype=torch.long,
                device=self.response_token_repair_up.weight.device,
            ),
            persistent=False,
        )
        # Optional executor-authored environment markers can hard-scope the
        # learned repair residual to one tool family. Empty is the exact legacy
        # behavior; no host-side parser or forced token is involved.
        self.response_token_repair_scope_environment_token_ids: tuple[int, ...] = ()
        # Default-off policy for a repair that may alter only response onset.
        self.response_token_repair_scope_until_first_semantic_token = False
        # Optional model-internal scalar calibration for selected rows on the
        # first semantic response token. Empty is exactly behavior-neutral.
        self.response_token_repair_onset_row_biases: tuple[
            tuple[int, float], ...
        ] = ()
        # Learned repair rows that may act at response onset but must be zero
        # after the first emitted semantic token. Empty is migration-exact.
        self.response_token_repair_post_onset_disabled_rows: tuple[int, ...] = ()
        self.register_buffer(
            "response_token_repair_post_onset_disabled_rows_tensor",
            torch.empty(
                0,
                dtype=torch.long,
                device=self.response_token_repair_up.weight.device,
            ),
            persistent=False,
        )
        # Optional scalar calibration after the first semantic response token.
        # It composes with onset-only and broad learned repair policies.
        self.response_token_repair_post_onset_row_biases: tuple[
            tuple[int, float], ...
        ] = ()
        # Checkpoint-gated live-generator latch for the semantic token that was
        # actually emitted. Old onset-policy checkpoints retain their original
        # behavior unless they explicitly opt into this corrected contract.
        self.response_token_repair_onset_generated_semantic_latch = False
        # A second, independently trainable low-rank residual handles native
        # sentence continuation after the first result-grounded semantic token.
        # Its output projection is exact-zero and the whole optional branch is
        # checkpoint-disabled by default, so every existing checkpoint remains
        # bit-identical. It deliberately shares the executor-authored TIMER
        # scope and causal semantic-onset clock with the established repair.
        self.response_token_continuation_norm = torch.nn.LayerNorm(
            response_temporal_dim,
            elementwise_affine=False,
            **lane_factory_kwargs,
        )
        self.response_token_continuation_down = torch.nn.Linear(
            response_temporal_dim,
            response_token_continuation_rank,
            bias=False,
            **lane_factory_kwargs,
        )
        self.response_token_continuation_up = torch.nn.Linear(
            response_token_continuation_rank,
            self.text_linear.weight.shape[0],
            bias=False,
            **lane_factory_kwargs,
        )
        self.response_token_continuation_conditioning_enabled = False

        self.action_depth_in = torch.nn.Linear(
            self.dim, action_depth_dim, bias=False, **lane_factory_kwargs
        )
        # A small causal recurrent adapter lets the native action lane retain
        # utterance-level evidence instead of classifying one final backbone
        # frame in isolation.  Its state is explicit in streaming inference.
        self.action_temporal_in = torch.nn.Linear(
            self.dim, action_depth_dim, bias=False, **lane_factory_kwargs
        )
        self.action_temporal_gru = torch.nn.GRU(
            action_depth_dim,
            action_depth_dim,
            num_layers=1,
            batch_first=True,
            device=self.text_linear.weight.device,
            dtype=self.text_linear.weight.dtype,
        )
        # The CALL_BEGIN gate owns its recurrent state.  Packet-learning
        # gradients therefore cannot turn a good timing/no-tool classifier
        # into an overcaller, and gate gradients cannot corrupt packet syntax.
        self.action_call_gate_in = torch.nn.Linear(
            self.dim, action_depth_dim, bias=False, **lane_factory_kwargs
        )
        self.action_call_gate_text_in = torch.nn.Linear(
            self.dim, action_depth_dim, bias=False, **lane_factory_kwargs
        )
        self.action_call_gate_gru = torch.nn.GRU(
            action_depth_dim,
            action_depth_dim,
            num_layers=1,
            batch_first=True,
            device=self.text_linear.weight.device,
            dtype=self.text_linear.weight.dtype,
        )
        self.action_call_gate_phase_emb = self.EmbeddingFactory(2, action_depth_dim)
        self.action_phase_emb = self.EmbeddingFactory(2, action_depth_dim)
        # A model-owned binary gate separates the causal decision to initiate a
        # call from within-packet token generation.  Zero logits intentionally
        # preserve legacy checkpoint behavior: CALL_BEGIN remains available at
        # the default zero threshold until a trained gate closes it.
        self.action_call_gate = torch.nn.Linear(
            action_depth_dim,
            1,
            bias=True,
            **lane_factory_kwargs,
        )
        torch.nn.init.zeros_(self.action_call_gate.weight)
        torch.nn.init.zeros_(self.action_call_gate.bias)
        # Tool intent and utterance endpoint are separate native decisions.
        # Intent gets a private causal state that accumulates both backbone and
        # PersonaPlex text-token evidence. The shared endpoint state remains a
        # residual so older factorized checkpoints retain their exact behavior
        # when these optional history modules are absent and zero-disabled.
        self.action_call_intent_in = torch.nn.Linear(
            self.dim, action_depth_dim, bias=False, **lane_factory_kwargs
        )
        self.action_call_intent_text_in = torch.nn.Linear(
            self.dim, action_depth_dim, bias=False, **lane_factory_kwargs
        )
        self.action_call_intent_norm = torch.nn.LayerNorm(
            action_depth_dim,
            device=self.text_linear.weight.device,
            dtype=self.text_linear.weight.dtype,
        )
        self.action_call_intent_gru = torch.nn.GRU(
            action_depth_dim,
            action_depth_dim,
            num_layers=1,
            batch_first=True,
            device=self.text_linear.weight.device,
            dtype=self.text_linear.weight.dtype,
        )
        self.action_call_intent_dropout = torch.nn.Dropout(p=0.15)
        # Zero initialization keeps old checkpoints open at threshold zero,
        # preserving their exact gate behavior until this head is trained.
        self.action_call_intent = torch.nn.Linear(
            action_depth_dim,
            1,
            bias=True,
            **lane_factory_kwargs,
        )
        torch.nn.init.zeros_(self.action_call_intent.weight)
        torch.nn.init.zeros_(self.action_call_intent.bias)
        # A low-capacity semantic residual consumes the expectation of the
        # frozen PersonaPlex text distribution rather than one sampled token.
        # Missing checkpoints zero-disable this residual, preserving legacy
        # factorized-gate behavior exactly.
        self.action_call_soft_intent_norm = torch.nn.LayerNorm(
            self.dim,
            device=self.text_linear.weight.device,
            dtype=self.text_linear.weight.dtype,
        )
        self.action_call_soft_intent = torch.nn.Linear(
            self.dim,
            1,
            bias=True,
            **lane_factory_kwargs,
        )
        torch.nn.init.zeros_(self.action_call_soft_intent.weight)
        torch.nn.init.zeros_(self.action_call_soft_intent.bias)
        self.action_depth_text_emb = self.EmbeddingFactory(
            self.text_card + 1, action_depth_dim
        )
        self.action_depth_emb = torch.nn.ModuleList(
            [
                self.EmbeddingFactory(self.action_card, action_depth_dim)
                for _ in range(action_slots - 1)
            ]
        )
        self.action_depth = StreamingTransformer(
            d_model=action_depth_dim,
            num_heads=action_depth_num_heads,
            num_layers=action_depth_num_layers,
            dim_feedforward=int(action_depth_hidden_scale * action_depth_dim),
            causal=True,
            context=None,
            positional_embedding="none",
            norm="rms_norm_f32",
            gating="silu",
            weights_per_step=action_slots,
            device=self.text_linear.weight.device,
            dtype=self.text_linear.weight.dtype,
        )
        self.action_depth.set_streaming_propagate(False)
        self.action_depth_linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    action_depth_dim,
                    self.action_card,
                    bias=False,
                    **lane_factory_kwargs,
                )
                for _ in range(action_slots)
            ]
        )
        # A zero-disabled, tool-gated residual predicts only TIMER's positional
        # duration token from the existing causal packet hidden state. It is
        # part of the neural action decoder, never a host-side parser. Because
        # it activates only after slot 1 is model-decoded as timer.create, HOME
        # and WEATHER packet logits remain byte-equivalent.
        self.action_timer_argument_linear = torch.nn.Linear(
            action_depth_dim,
            self.action_card,
            bias=False,
            **lane_factory_kwargs,
        )
        self.action_timer_argument_conditioning_enabled = False
        # A higher-capacity alternative reads the raw causal PersonaPlex state
        # at the call frame. Its zero output layer makes activation exact-no-op
        # until trained, while the TIMER/tool gate preserves all other tools.
        self.action_timer_argument_mlp_in = torch.nn.Linear(
            self.dim,
            action_depth_dim,
            bias=False,
            **lane_factory_kwargs,
        )
        self.action_timer_argument_mlp_norm = torch.nn.LayerNorm(
            action_depth_dim,
            **lane_factory_kwargs,
        )
        self.action_timer_argument_mlp_out = torch.nn.Linear(
            action_depth_dim,
            self.action_card,
            bias=False,
            **lane_factory_kwargs,
        )
        self.action_timer_argument_mlp_conditioning_enabled = False
        # Optional frozen decision branch. It reproduces a stable source
        # checkpoint's slot-0 logits on even CALL_BEGIN/NOOP frames while the
        # main packet decoder remains free to learn tool and argument content.
        # Legacy checkpoints omit these tensors and keep the branch disabled.
        self.action_start_depth_in = torch.nn.Linear(
            self.dim, action_depth_dim, bias=False, **lane_factory_kwargs
        )
        self.action_start_temporal_in = torch.nn.Linear(
            self.dim, action_depth_dim, bias=False, **lane_factory_kwargs
        )
        self.action_start_temporal_gru = torch.nn.GRU(
            action_depth_dim,
            action_depth_dim,
            num_layers=1,
            batch_first=True,
            device=self.text_linear.weight.device,
            dtype=self.text_linear.weight.dtype,
        )
        self.action_start_phase_emb = self.EmbeddingFactory(2, action_depth_dim)
        self.action_start_depth_text_emb = self.EmbeddingFactory(
            self.text_card + 1, action_depth_dim
        )
        self.action_start_depth = StreamingTransformer(
            d_model=action_depth_dim,
            num_heads=action_depth_num_heads,
            num_layers=action_depth_num_layers,
            dim_feedforward=int(action_depth_hidden_scale * action_depth_dim),
            causal=True,
            context=None,
            positional_embedding="none",
            norm="rms_norm_f32",
            gating="silu",
            weights_per_step=action_slots,
            device=self.text_linear.weight.device,
            dtype=self.text_linear.weight.dtype,
        )
        self.action_start_depth.set_streaming_propagate(False)
        self.action_start_linear = torch.nn.Linear(
            action_depth_dim,
            self.action_card,
            bias=False,
            **lane_factory_kwargs,
        )
        self.separate_action_start_branch_enabled = False
        depformer_dim = self.depformer_in[0].out_features
        self.action_summary_proj = torch.nn.Linear(
            self.dim, depformer_dim, bias=False, **lane_factory_kwargs
        )
        self.action_audio_conditioning_gate = torch.nn.Parameter(
            torch.zeros((), **lane_factory_kwargs)
        )

    @torch.no_grad()
    def copy_packet_decoder_to_start_branch(self) -> None:
        """Initialize the optional start branch from the current packet decoder."""

        self.action_start_depth_in.load_state_dict(self.action_depth_in.state_dict())
        self.action_start_temporal_in.load_state_dict(
            self.action_temporal_in.state_dict()
        )
        self.action_start_temporal_gru.load_state_dict(
            self.action_temporal_gru.state_dict()
        )
        self.action_start_phase_emb.load_state_dict(self.action_phase_emb.state_dict())
        self.action_start_depth_text_emb.load_state_dict(
            self.action_depth_text_emb.state_dict()
        )
        self.action_start_depth.load_state_dict(self.action_depth.state_dict())
        self.action_start_linear.load_state_dict(
            self.action_depth_linears[0].state_dict()
        )

    @torch.no_grad()
    def initialize_action_timer_argument_conditioning(self) -> None:
        """Enable an exact-zero TIMER duration residual over packet slot 3."""

        self.action_timer_argument_linear.weight.zero_()
        self.action_timer_argument_conditioning_enabled = True

    @torch.no_grad()
    def initialize_action_timer_argument_mlp_conditioning(self) -> None:
        """Enable an exact-zero nonlinear TIMER duration residual."""

        self.action_timer_argument_mlp_in.reset_parameters()
        self.action_timer_argument_mlp_norm.reset_parameters()
        self.action_timer_argument_mlp_out.weight.zero_()
        self.action_timer_argument_mlp_conditioning_enabled = True

    @torch.no_grad()
    def initialize_response_environment_conditioning(self) -> None:
        """Seed isolated speech-result embeddings without changing live output."""

        for target, source in zip(
            self.response_environment_history_emb,
            self.environment_history_emb,
            strict=True,
        ):
            target.load_state_dict(source.state_dict())
        self.response_environment_scale.zero_()
        self.response_environment_conditioning_enabled = True

    @torch.no_grad()
    def initialize_response_temporal_conditioning(self) -> None:
        """Enable ordered result speech behind the existing exact-zero gate."""

        if not self.response_environment_conditioning_enabled:
            self.initialize_response_environment_conditioning()
        self.response_temporal_base_in.reset_parameters()
        self.response_temporal_context_in.reset_parameters()
        self.response_temporal_gru.reset_parameters()
        self.response_temporal_out.reset_parameters()
        self.response_environment_scale.zero_()
        self.response_temporal_conditioning_enabled = True

    @torch.no_grad()
    def initialize_response_action_conditioning(self) -> None:
        """Enable exact-zero awareness of the model's emitted call packet."""

        if not self.response_temporal_conditioning_enabled:
            raise ValueError(
                "response action conditioning requires temporal response conditioning"
            )
        self.response_action_scale.zero_()
        self.response_action_conditioning_enabled = True

    @torch.no_grad()
    def initialize_response_eos_conditioning(self) -> None:
        """Enable an exact-zero native EOS bias over the temporal response state."""

        if not self.response_temporal_conditioning_enabled:
            raise ValueError("response EOS requires trained temporal response conditioning")
        self.response_eos_head.weight.zero_()
        self.response_eos_head.bias.zero_()
        self.response_eos_conditioning_enabled = True

    @torch.no_grad()
    def initialize_response_token_conditioning(self) -> None:
        """Enable an exact-zero native text-logit residual after tool results."""

        if not self.response_temporal_conditioning_enabled:
            raise ValueError(
                "response token conditioning requires temporal response conditioning"
            )
        self.response_token_head.weight.zero_()
        self.response_token_conditioning_enabled = True

    @torch.no_grad()
    def initialize_response_token_context_conditioning(self) -> None:
        """Enable exact-zero causal language context for the result-token head."""

        if not self.response_token_conditioning_enabled:
            raise ValueError(
                "response token context requires response token conditioning"
            )
        if not self.response_temporal_conditioning_enabled:
            raise ValueError(
                "response token context requires temporal response conditioning"
            )
        self.response_token_context_in.weight.zero_()
        self.response_token_context_conditioning_enabled = True

    @torch.no_grad()
    def initialize_response_token_repair_conditioning(self) -> None:
        """Enable an exact-zero low-rank residual over causal result state."""

        if not self.response_token_conditioning_enabled:
            raise ValueError(
                "response token repair requires response token conditioning"
            )
        if not self.response_temporal_conditioning_enabled:
            raise ValueError(
                "response token repair requires temporal response conditioning"
            )
        self.response_token_repair_down.reset_parameters()
        self.response_token_repair_up.weight.zero_()
        self.response_token_repair_conditioning_enabled = True

    @torch.no_grad()
    def initialize_response_token_repair_gate_conditioning(self) -> None:
        """Enable an identity-initialized affine selector over repair state."""

        if not self.response_token_repair_conditioning_enabled:
            raise ValueError("response repair gate requires response token repair")
        self.response_token_repair_gate.weight.zero_()
        self.response_token_repair_gate.bias.zero_()
        self.response_token_repair_gate_conditioning_enabled = True

    @torch.no_grad()
    def initialize_response_token_continuation_conditioning(self) -> None:
        """Enable an exact-zero post-onset continuation residual."""

        if not self.response_token_repair_conditioning_enabled:
            raise ValueError(
                "response token continuation requires response token repair"
            )
        if not self.response_token_repair_scope_environment_token_ids:
            raise ValueError(
                "response token continuation requires an executor result scope"
            )
        self.response_token_continuation_down.reset_parameters()
        self.response_token_continuation_up.weight.zero_()
        self.response_token_continuation_conditioning_enabled = True

    def set_response_token_repair_nonnegative_rows(
        self, row_ids: Sequence[int]
    ) -> None:
        """Clamp selected repair-logit rows at zero without changing others."""

        rows = tuple(sorted({int(row_id) for row_id in row_ids}))
        if any(
            row_id < 0
            or row_id >= self.response_token_repair_up.weight.shape[0]
            for row_id in rows
        ):
            raise ValueError("response repair nonnegative row is out of range")
        if rows and not self.response_token_repair_conditioning_enabled:
            raise ValueError(
                "nonnegative repair rows require response token repair"
            )
        self.response_token_repair_nonnegative_rows = rows
        self.response_token_repair_nonnegative_rows_tensor = torch.tensor(
            rows,
            dtype=torch.long,
            device=self.response_token_repair_nonnegative_rows_tensor.device,
        )

    def set_response_token_repair_scope_environment_token_ids(
        self, token_ids: Sequence[int]
    ) -> None:
        """Activate response repair only for matching typed result events."""

        tokens = tuple(sorted({int(token_id) for token_id in token_ids}))
        if any(
            token_id < 0 or token_id >= self.environment_card
            for token_id in tokens
        ):
            raise ValueError("response repair scope token is out of range")
        if tokens and not self.response_token_repair_conditioning_enabled:
            raise ValueError(
                "response repair environment scope requires response token repair"
            )
        self.response_token_repair_scope_environment_token_ids = tokens

    def set_response_token_repair_scope_until_first_semantic_token(
        self, enabled: bool
    ) -> None:
        """Disable a scoped repair after the first native semantic token."""

        enabled = bool(enabled)
        if enabled and not self.response_token_repair_conditioning_enabled:
            raise ValueError("response repair onset scope requires token repair")
        if enabled and not self.response_token_repair_scope_environment_token_ids:
            raise ValueError(
                "response repair onset scope requires environment scope tokens"
            )
        self.response_token_repair_scope_until_first_semantic_token = enabled

    def set_response_token_repair_onset_row_biases(
        self, row_biases: Sequence[tuple[int, float]]
    ) -> None:
        """Calibrate selected logits only at scoped response onset."""

        pairs = tuple(
            sorted((int(row_id), float(bias)) for row_id, bias in row_biases)
        )
        rows = tuple(row_id for row_id, _ in pairs)
        if len(set(rows)) != len(rows):
            raise ValueError("response repair onset rows must be unique")
        if any(
            row_id < 0
            or row_id >= self.response_token_repair_up.weight.shape[0]
            for row_id in rows
        ):
            raise ValueError("response repair onset row is out of range")
        if any(not math.isfinite(bias) or bias == 0 for _, bias in pairs):
            raise ValueError("response repair onset bias must be finite and nonzero")
        if pairs and not self.response_token_repair_conditioning_enabled:
            raise ValueError("response repair onset bias requires token repair")
        if pairs and not self.response_token_repair_scope_environment_token_ids:
            raise ValueError(
                "response repair onset bias requires environment scope tokens"
            )
        self.response_token_repair_onset_row_biases = pairs

    def set_response_token_repair_onset_generated_semantic_latch(
        self, enabled: bool
    ) -> None:
        """Latch an emitted semantic token before the next generator step."""

        enabled = bool(enabled)
        if enabled and not self.response_token_repair_conditioning_enabled:
            raise ValueError("response repair onset latch requires token repair")
        if enabled and not self.response_token_repair_scope_environment_token_ids:
            raise ValueError(
                "response repair onset latch requires environment scope tokens"
            )
        if enabled and not (
            self.response_token_repair_scope_until_first_semantic_token
            or self.response_token_repair_onset_row_biases
        ):
            raise ValueError(
                "response repair onset latch requires an onset repair policy"
            )
        self.response_token_repair_onset_generated_semantic_latch = enabled

    def set_response_token_repair_post_onset_disabled_rows(
        self, rows: Sequence[int]
    ) -> None:
        """Restrict selected learned repair rows to response onset."""

        normalized = tuple(sorted(int(row_id) for row_id in rows))
        if len(normalized) != len(set(normalized)):
            raise ValueError("post-onset disabled repair rows must be unique")
        if any(row_id < 0 or row_id >= self.text_card for row_id in normalized):
            raise ValueError("post-onset disabled repair row is out of range")
        if normalized and not self.response_token_repair_conditioning_enabled:
            raise ValueError("post-onset repair rows require token repair")
        if normalized and not self.response_token_repair_scope_environment_token_ids:
            raise ValueError(
                "post-onset repair rows require environment scope tokens"
            )
        self.response_token_repair_post_onset_disabled_rows = normalized
        self.response_token_repair_post_onset_disabled_rows_tensor = torch.tensor(
            normalized,
            dtype=torch.long,
            device=self.response_token_repair_post_onset_disabled_rows_tensor.device,
        )

    def set_response_token_repair_post_onset_row_biases(
        self, row_biases: Sequence[tuple[int, float]]
    ) -> None:
        """Calibrate selected logits only after response onset."""

        pairs = tuple(
            sorted((int(row_id), float(bias)) for row_id, bias in row_biases)
        )
        rows = tuple(row_id for row_id, _ in pairs)
        if len(rows) != len(set(rows)):
            raise ValueError("post-onset repair bias rows must be unique")
        if any(row_id < 0 or row_id >= self.text_card for row_id in rows):
            raise ValueError("post-onset repair bias row is out of range")
        if any(not math.isfinite(bias) or bias == 0 for _, bias in pairs):
            raise ValueError(
                "post-onset repair bias must be finite and nonzero"
            )
        if pairs and not self.response_token_repair_conditioning_enabled:
            raise ValueError("post-onset repair bias requires token repair")
        if pairs and not self.response_token_repair_scope_environment_token_ids:
            raise ValueError(
                "post-onset repair bias requires environment scope tokens"
            )
        self.response_token_repair_post_onset_row_biases = pairs

    @property
    def micro_padding_token_id(self) -> int:
        return MICRO_PADDING_TOKEN_ID

    def _validate_micro_tokens(
        self,
        sequence: torch.Tensor,
        action_tokens: torch.Tensor,
        environment_tokens: torch.Tensor,
    ) -> None:
        if sequence.dim() != 3:
            raise ValueError("PersonaPlex sequence must be [batch, streams, steps]")
        batch, codebooks, steps = sequence.shape
        if codebooks != self.num_codebooks:
            raise ValueError("unexpected PersonaPlex codebook count")
        if action_tokens.shape != (batch, steps, self.action_slots):
            raise ValueError("actions must be [batch, steps, five microtokens]")
        if environment_tokens.shape != (batch, steps, self.environment_slots):
            raise ValueError("environment must be [batch, steps, five microtokens]")
        if (action_tokens < MICRO_PADDING_TOKEN_ID).any() or (
            action_tokens >= self.action_card
        ).any():
            raise ValueError("action microtoken is outside its vocabulary")
        if (environment_tokens < MICRO_PADDING_TOKEN_ID).any() or (
            environment_tokens >= self.environment_card
        ).any():
            raise ValueError("environment microtoken is outside its vocabulary")

    def validate_action_frame_suffix(self, action_tokens: torch.Tensor) -> None:
        """Reject semantic tokens after a terminal action or padding slot."""

        if action_tokens.shape[-1] != self.action_slots:
            raise ValueError("action frame must contain five microtokens")
        closed = torch.zeros(
            action_tokens.shape[:-1], dtype=torch.bool, device=action_tokens.device
        )
        for slot in range(self.action_slots):
            token = action_tokens[..., slot]
            if (closed & (token >= 0)).any():
                raise ValueError("semantic action token appears after a terminal/padding slot")
            terminal = token < 0
            for terminal_id in self.terminal_action_token_ids:
                terminal |= token == terminal_id
            closed |= terminal

    @staticmethod
    def _pool_slot_embeddings(
        tokens: torch.Tensor,
        embeddings: torch.nn.ModuleList,
    ) -> torch.Tensor:
        pooled: Optional[torch.Tensor] = None
        valid_count = (tokens >= 0).sum(dim=-1, keepdim=True).clamp(min=1)
        for slot, embedding in enumerate(embeddings):
            valid = tokens[..., slot] >= 0
            safe_tokens = tokens[..., slot].clamp_min(0)
            slot_embedding = embedding(safe_tokens)
            slot_embedding = torch.where(
                valid.unsqueeze(-1),
                slot_embedding,
                torch.zeros_like(slot_embedding),
            )
            pooled = slot_embedding if pooled is None else pooled + slot_embedding
        assert pooled is not None
        return pooled / valid_count.to(dtype=pooled.dtype).sqrt()

    def action_frame_embedding(self, action_tokens: torch.Tensor) -> torch.Tensor:
        if action_tokens.shape[-1] != self.action_slots:
            raise ValueError("action frame must contain five microtokens")
        return self._pool_slot_embeddings(action_tokens, self.action_history_emb)

    def environment_frame_embedding(self, environment_tokens: torch.Tensor) -> torch.Tensor:
        if environment_tokens.shape[-1] != self.environment_slots:
            raise ValueError("environment frame must contain five microtokens")
        return self._pool_slot_embeddings(
            environment_tokens, self.environment_history_emb
        )

    def action_audio_summary(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """Zero-gated action conditioning for future grounded-speech training."""

        summary = self.action_summary_proj(self.action_frame_embedding(action_tokens))
        return torch.tanh(self.action_audio_conditioning_gate) * summary

    def embed_micro_codes(
        self,
        sequence: torch.Tensor,
        action_tokens: torch.Tensor,
        environment_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if not in_cuda_graph():
            self._validate_micro_tokens(sequence, action_tokens, environment_tokens)
        # Keep the conversational stream byte-equivalent to stock PersonaPlex.
        # Typed context is added only to the action branch after the frozen
        # temporal backbone in ``forward_micro_codes``.
        return self.embed_codes(sequence)

    def forward_micro_codes(
        self,
        sequence: torch.Tensor,
        action_tokens: torch.Tensor,
        environment_tokens: torch.Tensor,
        response_environment_state: Optional[torch.Tensor] = None,
        response_temporal_state: Optional[torch.Tensor] = None,
        response_terminal_seen: Optional[torch.Tensor] = None,
        response_active: Optional[torch.Tensor] = None,
        response_token_repair_scope_seen: Optional[torch.Tensor] = None,
        response_token_repair_scope_semantic_seen: Optional[torch.Tensor] = None,
    ) -> AgentMicroBackboneOutput:
        embeddings = self.embed_micro_codes(sequence, action_tokens, environment_tokens)
        base_transformer_out = self.transformer(embeddings)
        if self.out_norm:
            base_transformer_out = self.out_norm(base_transformer_out)
        batch = base_transformer_out.shape[0]
        if response_environment_state is None:
            response_environment_state = torch.zeros(
                (batch, 1, self.dim),
                dtype=base_transformer_out.dtype,
                device=base_transformer_out.device,
            )
        elif response_environment_state.shape != (batch, 1, self.dim):
            raise ValueError("response environment state must be [batch,1,dim]")
        if response_temporal_state is None:
            response_temporal_state = torch.zeros(
                (self.response_temporal_gru.num_layers, batch, self.response_temporal_dim),
                dtype=base_transformer_out.dtype,
                device=base_transformer_out.device,
            )
        elif response_temporal_state.shape != (
            self.response_temporal_gru.num_layers,
            batch,
            self.response_temporal_dim,
        ):
            raise ValueError("response temporal state has an invalid shape")
        if response_terminal_seen is None:
            response_terminal_seen = torch.zeros(
                (batch, 1, 1), dtype=torch.bool, device=base_transformer_out.device
            )
        elif response_terminal_seen.shape != (batch, 1, 1):
            raise ValueError("response terminal-seen state must be [batch,1,1]")
        if response_active is None:
            response_active = torch.zeros(
                (batch, 1, 1), dtype=torch.bool, device=base_transformer_out.device
            )
        elif response_active.shape != (batch, 1, 1):
            raise ValueError("response active state must be [batch,1,1]")
        if response_token_repair_scope_seen is None:
            response_token_repair_scope_seen = torch.zeros(
                (batch, 1, 1), dtype=torch.bool, device=base_transformer_out.device
            )
        elif response_token_repair_scope_seen.shape != (batch, 1, 1):
            raise ValueError("response repair scope state must be [batch,1,1]")
        if response_token_repair_scope_semantic_seen is None:
            response_token_repair_scope_semantic_seen = torch.zeros(
                (batch, 1, 1), dtype=torch.bool, device=base_transformer_out.device
            )
        elif response_token_repair_scope_semantic_seen.shape != (batch, 1, 1):
            raise ValueError(
                "response repair scope semantic state must be [batch,1,1]"
            )
        speech_transformer_out = base_transformer_out
        next_response_environment_state = response_environment_state
        next_response_temporal_state = response_temporal_state
        next_response_terminal_seen = response_terminal_seen
        next_response_active = response_active
        next_response_token_repair_scope_seen = response_token_repair_scope_seen
        next_response_token_repair_scope_semantic_seen = (
            response_token_repair_scope_semantic_seen
        )
        response_eos_bias = torch.zeros(
            base_transformer_out.shape[:2],
            dtype=base_transformer_out.dtype,
            device=base_transformer_out.device,
        )
        response_token_bias: Optional[torch.Tensor] = None
        response_token_repair_bias: Optional[torch.Tensor] = None
        response_token_repair_hidden: Optional[torch.Tensor] = None
        response_token_repair_source: Optional[torch.Tensor] = None
        response_token_continuation_bias: Optional[torch.Tensor] = None
        response_token_continuation_hidden: Optional[torch.Tensor] = None
        response_token_continuation_source: Optional[torch.Tensor] = None
        if self.response_environment_conditioning_enabled:
            response_events = self._pool_slot_embeddings(
                environment_tokens,
                self.response_environment_history_emb,
            )
            if self.response_action_conditioning_enabled:
                # NOOP is protocol timing, not call semantics. Excluding it
                # prevents long conversations from accumulating a large
                # irrelevant response bias while retaining the complete typed
                # CALL_BEGIN/tool/ref/arguments/ACTION_END packet.
                response_action_tokens = action_tokens.masked_fill(
                    action_tokens == self.no_action_token_id,
                    MICRO_PADDING_TOKEN_ID,
                )
                response_action_events = self._pool_slot_embeddings(
                    response_action_tokens,
                    self.action_history_emb,
                ).detach()
                response_events = response_events + response_action_events * torch.tanh(
                    self.response_action_scale
                ).view(1, 1, -1)
            cumulative_response = (
                torch.cumsum(response_events, dim=1)
                + response_environment_state
            )
            next_response_environment_state = cumulative_response[:, -1:]
            scale = torch.tanh(self.response_environment_scale).view(1, 1, -1)
            if self.response_temporal_conditioning_enabled:
                terminal_marker = torch.zeros(
                    environment_tokens.shape[:2],
                    dtype=torch.bool,
                    device=environment_tokens.device,
                )
                for token_id in self.response_terminal_status_token_ids:
                    terminal_marker |= (environment_tokens == token_id).any(dim=-1)
                terminal_seen = (
                    torch.cumsum(terminal_marker.to(dtype=torch.int32), dim=1) > 0
                ) | response_terminal_seen[:, :, 0]
                end_marker = (
                    environment_tokens == self.response_obs_end_token_id
                ).any(dim=-1)
                completion = terminal_seen & end_marker
                active = (
                    torch.cumsum(completion.to(dtype=torch.int32), dim=1) > 0
                ) | response_active[:, :, 0]
                repair_active = active
                repair_onset_active = torch.zeros_like(active)
                repair_post_onset_active = torch.zeros_like(active)
                if self.response_token_repair_scope_environment_token_ids:
                    scope_marker = torch.zeros_like(active)
                    for token_id in (
                        self.response_token_repair_scope_environment_token_ids
                    ):
                        scope_marker |= (environment_tokens == token_id).any(
                            dim=-1
                        )
                    begin_marker = (
                        environment_tokens == self.response_obs_begin_token_id
                    ).any(dim=-1)
                    match_count = torch.cumsum(
                        scope_marker.to(dtype=torch.int32), dim=1
                    )
                    count_before_frame = match_count - scope_marker.to(
                        dtype=torch.int32
                    )
                    begin_match_base = torch.where(
                        begin_marker,
                        count_before_frame,
                        torch.zeros_like(count_before_frame),
                    )
                    last_begin_match_base = torch.cummax(
                        begin_match_base, dim=1
                    ).values
                    has_begin = (
                        torch.cumsum(begin_marker.to(dtype=torch.int32), dim=1)
                        > 0
                    )
                    scope_seen = torch.where(
                        has_begin,
                        match_count > last_begin_match_base,
                        response_token_repair_scope_seen[:, :, 0]
                        | (match_count > 0),
                    )
                    repair_active = active & scope_seen
                    repair_onset_active = repair_active
                    next_response_token_repair_scope_seen = scope_seen[
                        :, -1:, None
                    ]
                    if (
                        self.response_token_repair_scope_until_first_semantic_token
                        or self.response_token_repair_onset_row_biases
                        or self.response_token_repair_post_onset_disabled_rows
                        or self.response_token_repair_post_onset_row_biases
                        or self.response_token_continuation_conditioning_enabled
                    ):
                        # ``sequence[:, 0]`` is the causal text input. In
                        # streaming it contains the token sampled on the prior
                        # frame, so including the current input marker keeps the
                        # repair active for the first token and disables it for
                        # every token that follows.
                        semantic_marker = (
                            (sequence[:, 0] >= 4) & active & scope_seen
                        )
                        semantic_count = torch.cumsum(
                            semantic_marker.to(dtype=torch.int32), dim=1
                        )
                        semantic_before_frame = (
                            semantic_count
                            - semantic_marker.to(dtype=torch.int32)
                        )
                        begin_semantic_base = torch.where(
                            begin_marker,
                            semantic_before_frame,
                            torch.zeros_like(semantic_before_frame),
                        )
                        last_begin_semantic_base = torch.cummax(
                            begin_semantic_base, dim=1
                        ).values
                        semantic_seen = torch.where(
                            has_begin,
                            semantic_count > last_begin_semantic_base,
                            response_token_repair_scope_semantic_seen[:, :, 0]
                            | (semantic_count > 0),
                        )
                        repair_post_onset_active = repair_active & semantic_seen
                        repair_onset_active = repair_active & ~semantic_seen
                        if (
                            self.response_token_repair_scope_until_first_semantic_token
                        ):
                            repair_active = repair_onset_active
                        next_response_token_repair_scope_semantic_seen = (
                            semantic_seen[:, -1:, None]
                        )
                temporal_input = (
                    self.response_temporal_base_in(base_transformer_out.detach())
                    + self.response_temporal_context_in(cumulative_response)
                )
                temporal_output, next_response_temporal_state = (
                    self.response_temporal_gru(
                        temporal_input,
                        response_temporal_state,
                    )
                )
                response_update = self.response_temporal_out(temporal_output)
                speech_transformer_out = (
                    base_transformer_out
                    + response_update
                    * scale
                    * active[:, :, None].to(dtype=response_update.dtype)
                )
                next_response_terminal_seen = terminal_seen[:, -1:, None]
                next_response_active = active[:, -1:, None]
                if self.response_eos_conditioning_enabled:
                    response_eos_bias = (
                        self.response_eos_head(temporal_output).squeeze(-1)
                        * active.to(dtype=temporal_output.dtype)
                    )
                if self.response_token_conditioning_enabled:
                    response_token_features = temporal_output
                    if self.response_token_context_conditioning_enabled:
                        response_token_features = (
                            response_token_features
                            + self.response_token_context_in(
                                base_transformer_out.detach()
                            )
                        )
                    response_token_bias = self.response_token_head(
                        response_token_features
                    ) * active[:, :, None].to(dtype=temporal_output.dtype)
                    if self.response_token_repair_conditioning_enabled:
                        # The source response path is frozen for this profile.
                        # Detaching its features avoids retaining gradients
                        # through PersonaPlex while still giving the repair
                        # lane the complete causal result/text state.
                        repair_features = response_token_features.detach()
                        repair_source = self.response_token_repair_norm(
                            repair_features
                        )
                        repair_hidden = torch.nn.functional.silu(
                            self.response_token_repair_down(
                                repair_source
                            )
                        )
                        response_token_repair_source = repair_source * repair_active[
                            :, :, None
                        ].to(dtype=repair_source.dtype)
                        response_token_repair_hidden = repair_hidden * repair_active[
                            :, :, None
                        ].to(dtype=repair_hidden.dtype)
                        repair_output = self.response_token_repair_up(
                            repair_hidden
                        )
                        if self.response_token_repair_gate_conditioning_enabled:
                            repair_gate = 1 + 0.5 * torch.tanh(
                                self.response_token_repair_gate(repair_source)
                            )
                            repair_output = repair_output * repair_gate
                        if self.response_token_repair_nonnegative_rows:
                            constrained_rows = (
                                self.response_token_repair_nonnegative_rows_tensor
                            )
                            constrained_output = repair_output.index_select(
                                -1, constrained_rows
                            ).clamp_min(0)
                            repair_output = repair_output.clone()
                            repair_output.index_copy_(
                                -1, constrained_rows, constrained_output
                            )
                        response_token_repair_bias = repair_output * repair_active[
                            :, :, None
                        ].to(dtype=repair_hidden.dtype)
                        if self.response_token_repair_post_onset_disabled_rows:
                            onset_rows = (
                                self.response_token_repair_post_onset_disabled_rows_tensor
                            )
                            onset_only_output = response_token_repair_bias.index_select(
                                -1, onset_rows
                            ) * repair_onset_active[:, :, None].to(
                                dtype=repair_hidden.dtype
                            )
                            response_token_repair_bias = (
                                response_token_repair_bias.clone()
                            )
                            response_token_repair_bias.index_copy_(
                                -1, onset_rows, onset_only_output
                            )
                        if self.response_token_repair_onset_row_biases:
                            onset_bias = torch.zeros_like(repair_output)
                            for row_id, bias in (
                                self.response_token_repair_onset_row_biases
                            ):
                                onset_bias[..., row_id] = bias
                            response_token_repair_bias = (
                                response_token_repair_bias
                                + onset_bias
                                * repair_onset_active[:, :, None].to(
                                    dtype=repair_hidden.dtype
                                )
                            )
                        if self.response_token_repair_post_onset_row_biases:
                            post_onset_bias = torch.zeros_like(repair_output)
                            for row_id, bias in (
                                self.response_token_repair_post_onset_row_biases
                            ):
                                post_onset_bias[..., row_id] = bias
                            response_token_repair_bias = (
                                response_token_repair_bias
                                + post_onset_bias
                                * repair_post_onset_active[:, :, None].to(
                                    dtype=repair_hidden.dtype
                                )
                            )
                        response_token_bias = (
                            response_token_bias + response_token_repair_bias
                        )
                        if self.response_token_continuation_conditioning_enabled:
                            continuation_source = (
                                self.response_token_continuation_norm(
                                    repair_features
                                )
                            )
                            continuation_hidden = torch.nn.functional.silu(
                                self.response_token_continuation_down(
                                    continuation_source
                                )
                            )
                            continuation_active = repair_post_onset_active[
                                :, :, None
                            ].to(dtype=continuation_hidden.dtype)
                            response_token_continuation_source = (
                                continuation_source * continuation_active
                            )
                            response_token_continuation_hidden = (
                                continuation_hidden * continuation_active
                            )
                            response_token_continuation_bias = (
                                self.response_token_continuation_up(
                                    continuation_hidden
                                )
                                * continuation_active
                            )
                            response_token_bias = (
                                response_token_bias
                                + response_token_continuation_bias
                            )
            else:
                # Backward-compatible v45-v50 static response checkpoints.
                speech_transformer_out = (
                    base_transformer_out + cumulative_response * scale
                )
        action_input = (
            base_transformer_out.detach()
            if self.action_backbone_adapter_detach_input
            else base_transformer_out
        )
        action_input = (
            action_input
            + self.action_frame_embedding(action_tokens)
            + self.environment_frame_embedding(environment_tokens)
        )
        transformer_out = self.action_backbone_adapter(action_input)
        assert isinstance(transformer_out, torch.Tensor)
        text_logits = self.text_linear(speech_transformer_out)
        if response_token_bias is not None:
            text_logits = text_logits + response_token_bias
        if self.response_eos_conditioning_enabled:
            if not self.response_temporal_conditioning_enabled:
                raise RuntimeError("response EOS is enabled without temporal conditioning")
            text_logits = text_logits.clone()
            text_logits[..., self.response_eos_token_id] += response_eos_bias
        return AgentMicroBackboneOutput(
            transformer_out=transformer_out,
            base_transformer_out=base_transformer_out,
            speech_transformer_out=speech_transformer_out,
            response_environment_state=next_response_environment_state,
            response_temporal_state=next_response_temporal_state,
            response_terminal_seen=next_response_terminal_seen,
            response_active=next_response_active,
            response_token_repair_scope_seen=(
                next_response_token_repair_scope_seen
            ),
            response_token_repair_scope_semantic_seen=(
                next_response_token_repair_scope_semantic_seen
            ),
            text_logits=text_logits[:, None],
            response_token_repair_bias=response_token_repair_bias,
            response_token_repair_hidden=response_token_repair_hidden,
            response_token_repair_source=response_token_repair_source,
            response_token_continuation_bias=(
                response_token_continuation_bias
            ),
            response_token_continuation_hidden=(
                response_token_continuation_hidden
            ),
            response_token_continuation_source=(
                response_token_continuation_source
            ),
        )

    def forward_action_start_training(
        self,
        transformer_out: torch.Tensor,
        text_tokens: torch.Tensor,
        *,
        frame_phase: torch.Tensor,
    ) -> torch.Tensor:
        """Return frozen-source slot-0 logits for independent cached training."""

        batch, steps, _ = transformer_out.shape
        if text_tokens.shape != (batch, steps) or frame_phase.shape != (batch, steps):
            raise ValueError("start-branch text/phase must align with backbone states")
        temporal_output, _ = self.action_start_temporal_gru(
            self.action_start_temporal_in(transformer_out)
        )
        condition = (
            self.action_start_depth_in(transformer_out)
            + temporal_output
            + self.action_start_phase_emb(frame_phase)
            + self.action_start_depth_text_emb(text_tokens)
        )
        depth_input = condition.view(batch * steps, 1, self.action_depth_dim)
        depth_output = self.action_start_depth(depth_input)
        return self.action_start_linear(depth_output[:, 0]).view(
            batch, steps, self.action_card
        )

    def forward_action_depth_gate_and_intent_training(
        self,
        transformer_out: torch.Tensor,
        text_tokens: torch.Tensor,
        action_targets: torch.Tensor,
        frame_phase: Optional[torch.Tensor] = None,
        soft_text_token_ids: Optional[torch.Tensor] = None,
        soft_text_token_weights: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-forced packet, endpoint, and semantic-intent logits."""

        batch, steps, _ = transformer_out.shape
        if text_tokens.shape != (batch, steps):
            raise ValueError("text targets must align with action frames")
        if action_targets.shape != (batch, steps, self.action_slots):
            raise ValueError("action targets must align with five frame slots")
        if frame_phase is None:
            frame_phase = torch.arange(steps, device=transformer_out.device).view(1, -1)
            frame_phase = (frame_phase % 2).expand(batch, -1)
        if frame_phase.shape != (batch, steps):
            raise ValueError("action packet phase must be [batch, steps]")
        if (frame_phase < 0).any() or (frame_phase > 1).any():
            raise ValueError("action packet phase must be even=0 or odd=1")
        call_gate_logits, call_intent_logits = (
            self.forward_call_gate_and_intent_training(
                transformer_out,
                text_tokens,
                frame_phase=frame_phase,
                soft_text_token_ids=soft_text_token_ids,
                soft_text_token_weights=soft_text_token_weights,
            )
        )
        temporal_output, _ = self.action_temporal_gru(
            self.action_temporal_in(transformer_out)
        )
        condition = (
            self.action_depth_in(transformer_out)
            + temporal_output
            + self.action_phase_emb(frame_phase)
        )
        depth_inputs: list[torch.Tensor] = []
        for slot in range(self.action_slots):
            if slot == 0:
                previous_embedding = self.action_depth_text_emb(text_tokens)
            else:
                previous_embedding = self.action_depth_emb[slot - 1](
                    action_targets[:, :, slot - 1]
                )
            depth_inputs.append(condition + previous_embedding)
        depth_input = torch.stack(depth_inputs, dim=2).view(
            batch * steps, self.action_slots, self.action_depth_dim
        )
        depth_output = self.action_depth(depth_input)
        logits = [
            self.action_depth_linears[slot](depth_output[:, slot]).view(
                batch, steps, self.action_card
            )
            for slot in range(self.action_slots)
        ]
        stacked_logits = torch.stack(logits, dim=2)
        if self.action_timer_argument_conditioning_enabled:
            timer_rows = action_targets[:, :, 1] == self.action_timer_tool_token_id
            timer_argument_logits = self.action_timer_argument_linear(
                depth_output[:, 3]
            ).view(batch, steps, self.action_card)
            stacked_logits = stacked_logits.clone()
            stacked_logits[:, :, 3] = torch.where(
                timer_rows.unsqueeze(-1),
                stacked_logits[:, :, 3] + timer_argument_logits,
                stacked_logits[:, :, 3],
            )
        if self.action_timer_argument_mlp_conditioning_enabled:
            timer_rows = action_targets[:, :, 1] == self.action_timer_tool_token_id
            timer_argument_hidden = torch.nn.functional.silu(
                self.action_timer_argument_mlp_norm(
                    self.action_timer_argument_mlp_in(transformer_out)
                )
            )
            timer_argument_logits = self.action_timer_argument_mlp_out(
                timer_argument_hidden
            )
            stacked_logits = stacked_logits.clone()
            stacked_logits[:, :, 3] = torch.where(
                timer_rows.unsqueeze(-1),
                stacked_logits[:, :, 3] + timer_argument_logits,
                stacked_logits[:, :, 3],
            )
        if self.separate_action_start_branch_enabled:
            start_logits = self.forward_action_start_training(
                transformer_out,
                text_tokens,
                frame_phase=frame_phase,
            )
            stacked_logits = stacked_logits.clone()
            stacked_logits[:, :, 0] = torch.where(
                (frame_phase == 0).unsqueeze(-1),
                start_logits,
                stacked_logits[:, :, 0],
            )
        return stacked_logits, call_gate_logits, call_intent_logits

    def forward_call_gate_and_intent_training(
        self,
        transformer_out: torch.Tensor,
        text_tokens: torch.Tensor,
        frame_phase: Optional[torch.Tensor] = None,
        soft_text_token_ids: Optional[torch.Tensor] = None,
        soft_text_token_weights: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced endpoint and semantic intent without packet/audio work."""

        batch, steps, _ = transformer_out.shape
        if text_tokens.shape != (batch, steps):
            raise ValueError("text targets must align with action frames")
        if frame_phase is None:
            frame_phase = torch.arange(steps, device=transformer_out.device).view(1, -1)
            frame_phase = (frame_phase % 2).expand(batch, -1)
        if frame_phase.shape != (batch, steps):
            raise ValueError("action packet phase must be [batch, steps]")
        if (frame_phase < 0).any() or (frame_phase > 1).any():
            raise ValueError("action packet phase must be even=0 or odd=1")
        text_embeddings = self.text_emb(text_tokens).detach()
        call_gate_temporal_output, _ = self.action_call_gate_gru(
            self.action_call_gate_in(transformer_out)
        )
        call_gate_condition = (
            call_gate_temporal_output
            + self.action_call_gate_text_in(text_embeddings)
            + self.action_call_gate_phase_emb(frame_phase)
        )
        call_intent_input = self.action_call_intent_norm(
            self.action_call_intent_in(transformer_out)
            + self.action_call_intent_text_in(text_embeddings)
        )
        call_intent_temporal_output, _ = self.action_call_intent_gru(
            self.action_call_intent_dropout(call_intent_input)
        )
        call_gate_logits = self.action_call_gate(call_gate_condition).squeeze(-1)
        hard_intent_logits = self.action_call_intent(
            call_gate_condition + call_intent_temporal_output
        ).squeeze(-1)
        soft_text = self._frozen_soft_text_embedding(
            transformer_out,
            cached_token_ids=soft_text_token_ids,
            cached_token_weights=soft_text_token_weights,
        )
        soft_intent_logits = self.action_call_soft_intent(
            self.action_call_soft_intent_norm(soft_text)
        ).squeeze(-1)
        call_intent_logits = hard_intent_logits + soft_intent_logits
        return call_gate_logits, call_intent_logits

    def _frozen_soft_text_embedding(
        self,
        transformer_out: torch.Tensor,
        *,
        text_logits: Optional[torch.Tensor] = None,
        cached_token_ids: Optional[torch.Tensor] = None,
        cached_token_weights: Optional[torch.Tensor] = None,
        top_k: int = 32,
    ) -> torch.Tensor:
        """Return a top-k expectation from the frozen PersonaPlex text head."""

        with torch.no_grad():
            if (cached_token_ids is None) != (cached_token_weights is None):
                raise ValueError("soft text cache requires IDs and weights together")
            if cached_token_ids is not None and cached_token_weights is not None:
                if cached_token_ids.shape != cached_token_weights.shape:
                    raise ValueError("soft text cache IDs and weights must align")
                if cached_token_ids.shape[:-1] != transformer_out.shape[:-1]:
                    raise ValueError("soft text cache must align with backbone frames")
                text_vocabulary = self.text_linear.weight.shape[0]
                if (cached_token_ids < 0).any() or (
                    cached_token_ids >= text_vocabulary
                ).any():
                    raise ValueError("soft text cache token is outside the vocabulary")
                weights = cached_token_weights.to(dtype=transformer_out.dtype)
                if not torch.isfinite(weights).all() or (weights < 0).any():
                    raise ValueError("soft text cache weights are invalid")
                embeddings = self.text_emb(cached_token_ids)
                return (embeddings * weights.unsqueeze(-1)).sum(dim=-2)
            if text_logits is None:
                text_logits = self.text_linear(transformer_out.detach())
            if text_logits.shape[:-1] != transformer_out.shape[:-1]:
                raise ValueError("soft text logits must align with backbone frames")
            text_vocabulary = self.text_linear.weight.shape[0]
            if text_logits.shape[-1] != text_vocabulary:
                raise ValueError("soft text logits use the wrong vocabulary")
            count = min(top_k, text_vocabulary)
            values, indices = torch.topk(text_logits, k=count, dim=-1)
            weights = torch.softmax(values.float(), dim=-1).to(
                dtype=transformer_out.dtype
            )
            embeddings = self.text_emb(indices)
            return (embeddings * weights.unsqueeze(-1)).sum(dim=-2)

    def forward_action_depth_and_gate_training(
        self,
        transformer_out: torch.Tensor,
        text_tokens: torch.Tensor,
        action_targets: torch.Tensor,
        frame_phase: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backward-compatible packet and endpoint-gate training interface."""

        logits, endpoint_logits, _intent_logits = (
            self.forward_action_depth_gate_and_intent_training(
                transformer_out,
                text_tokens,
                action_targets,
                frame_phase=frame_phase,
            )
        )
        return logits, endpoint_logits

    def forward_action_depth_training(
        self,
        transformer_out: torch.Tensor,
        text_tokens: torch.Tensor,
        action_targets: torch.Tensor,
        frame_phase: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Teacher-forced within-frame action autoregression."""

        logits, _ = self.forward_action_depth_and_gate_training(
            transformer_out,
            text_tokens,
            action_targets,
            frame_phase=frame_phase,
        )
        return logits

    @torch.no_grad()
    def sample_action_frame(
        self,
        transformer_out: torch.Tensor,
        text_token: torch.Tensor,
        *,
        forced_tokens: Optional[torch.Tensor] = None,
        forced_mask: Optional[torch.Tensor] = None,
        temporal_state: Optional[torch.Tensor] = None,
        start_temporal_state: Optional[torch.Tensor] = None,
        call_gate_temporal_state: Optional[torch.Tensor] = None,
        call_intent_temporal_state: Optional[torch.Tensor] = None,
        text_logits: Optional[torch.Tensor] = None,
        soft_text_token_ids: Optional[torch.Tensor] = None,
        soft_text_token_weights: Optional[torch.Tensor] = None,
        frame_phase: int | torch.Tensor = 0,
        logit_masker: Optional[ActionLogitMasker] = None,
        use_sampling: bool = True,
        temperature: float = 0.7,
        top_k: int = 25,
    ) -> ActionMicroOutput:
        """Generate five ordered action slots; padding is never sampled."""

        batch, steps, _ = transformer_out.shape
        if steps != 1 or text_token.shape != (batch,):
            raise ValueError("streaming action decode requires one temporal step")
        if forced_tokens is not None and forced_tokens.shape != (batch, self.action_slots):
            raise ValueError("forced action tokens must be [batch, five]")
        if forced_mask is not None and forced_mask.shape != (batch, self.action_slots):
            raise ValueError("forced action mask must be [batch, five]")
        if forced_mask is not None and forced_tokens is None:
            raise ValueError("a forced mask requires forced tokens")
        if forced_tokens is not None:
            if (forced_tokens < MICRO_PADDING_TOKEN_ID).any() or (
                forced_tokens >= self.action_card
            ).any():
                raise ValueError("forced action microtoken is outside its vocabulary")
            if forced_mask is None:
                forced_mask = forced_tokens >= 0
            else:
                forced_mask = forced_mask.to(dtype=torch.bool)
            forced_tokens = torch.where(
                forced_mask,
                forced_tokens,
                torch.full_like(forced_tokens, MICRO_PADDING_TOKEN_ID),
            )
            self.validate_action_frame_suffix(forced_tokens)

        if isinstance(frame_phase, int):
            phase_tensor = torch.full(
                (batch,), frame_phase, dtype=torch.long, device=transformer_out.device
            )
        else:
            phase_tensor = frame_phase.to(
                device=transformer_out.device, dtype=torch.long
            )
        if phase_tensor.shape != (batch,) or (phase_tensor < 0).any() or (
            phase_tensor > 1
        ).any():
            raise ValueError("streaming action phase must be [batch] with values 0 or 1")

        tokens = torch.full(
            (batch, self.action_slots),
            MICRO_PADDING_TOKEN_ID,
            dtype=torch.long,
            device=transformer_out.device,
        )
        active = torch.ones(batch, dtype=torch.bool, device=transformer_out.device)
        if temporal_state is not None and temporal_state.shape != (
            self.action_temporal_gru.num_layers,
            batch,
            self.action_depth_dim,
        ):
            raise ValueError("action temporal state has the wrong shape")
        if start_temporal_state is not None and start_temporal_state.shape != (
            self.action_start_temporal_gru.num_layers,
            batch,
            self.action_depth_dim,
        ):
            raise ValueError("action start temporal state has the wrong shape")
        if call_gate_temporal_state is not None and call_gate_temporal_state.shape != (
            self.action_call_gate_gru.num_layers,
            batch,
            self.action_depth_dim,
        ):
            raise ValueError("action call-gate temporal state has the wrong shape")
        if call_intent_temporal_state is not None and call_intent_temporal_state.shape != (
            self.action_call_intent_gru.num_layers,
            batch,
            self.action_depth_dim,
        ):
            raise ValueError("action call-intent temporal state has the wrong shape")
        temporal_output, next_temporal_state = self.action_temporal_gru(
            self.action_temporal_in(transformer_out), temporal_state
        )
        condition = (
            self.action_depth_in(transformer_out)
            + temporal_output
            + self.action_phase_emb(phase_tensor)[:, None]
        )
        start_decision_logits: Optional[torch.Tensor] = None
        next_start_temporal_state = start_temporal_state
        if self.separate_action_start_branch_enabled:
            start_temporal_output, next_start_temporal_state = (
                self.action_start_temporal_gru(
                    self.action_start_temporal_in(transformer_out),
                    start_temporal_state,
                )
            )
            start_condition = (
                self.action_start_depth_in(transformer_out)
                + start_temporal_output
                + self.action_start_phase_emb(phase_tensor)[:, None]
                + self.action_start_depth_text_emb(text_token)[:, None]
            )
            assert not self.action_start_depth.is_streaming
            with self.action_start_depth.streaming(batch):
                start_depth_output = self.action_start_depth(start_condition)
            start_decision_logits = self.action_start_linear(
                start_depth_output
            )[:, 0].float()
        call_gate_temporal_output, next_call_gate_temporal_state = (
            self.action_call_gate_gru(
                self.action_call_gate_in(transformer_out),
                call_gate_temporal_state,
            )
        )
        call_gate_condition = (
            call_gate_temporal_output[:, 0]
            + self.action_call_gate_text_in(self.text_emb(text_token).detach())
            + self.action_call_gate_phase_emb(phase_tensor)
        )
        call_intent_input = self.action_call_intent_norm(
            self.action_call_intent_in(transformer_out)
            + self.action_call_intent_text_in(self.text_emb(text_token).detach())[:, None]
        )
        call_intent_temporal_output, next_call_intent_temporal_state = (
            self.action_call_intent_gru(
                call_intent_input,
                call_intent_temporal_state,
            )
        )
        call_endpoint_logits = self.action_call_gate(call_gate_condition).squeeze(-1).float()
        hard_intent_logits = self.action_call_intent(
            call_gate_condition + call_intent_temporal_output[:, 0]
        ).squeeze(-1).float()
        if soft_text_token_ids is not None and soft_text_token_ids.ndim == 2:
            soft_text_token_ids = soft_text_token_ids[:, None]
        if soft_text_token_weights is not None and soft_text_token_weights.ndim == 2:
            soft_text_token_weights = soft_text_token_weights[:, None]
        soft_text = self._frozen_soft_text_embedding(
            transformer_out,
            text_logits=text_logits,
            cached_token_ids=soft_text_token_ids,
            cached_token_weights=soft_text_token_weights,
        )[:, 0]
        call_soft_intent_logits = self.action_call_soft_intent(
            self.action_call_soft_intent_norm(soft_text)
        ).squeeze(-1).float()
        call_intent_logits = hard_intent_logits + call_soft_intent_logits
        call_gate_logits = torch.minimum(call_endpoint_logits, call_intent_logits)
        all_logits: list[torch.Tensor] = []
        assert not self.action_depth.is_streaming
        with self.action_depth.streaming(batch):
            for slot in range(self.action_slots):
                if slot == 0:
                    previous_embedding = self.action_depth_text_emb(text_token)[:, None]
                else:
                    previous_embedding = self.action_depth_emb[slot - 1](
                        tokens[:, slot - 1]
                    )[:, None]
                depth_output = self.action_depth(condition + previous_embedding)
                logits = self.action_depth_linears[slot](depth_output)[:, 0].float()
                if slot == 3 and self.action_timer_argument_conditioning_enabled:
                    timer_rows = tokens[:, 1] == self.action_timer_tool_token_id
                    timer_argument_logits = self.action_timer_argument_linear(
                        depth_output[:, 0]
                    ).float()
                    logits = torch.where(
                        timer_rows.unsqueeze(-1),
                        logits + timer_argument_logits,
                        logits,
                    )
                if slot == 3 and self.action_timer_argument_mlp_conditioning_enabled:
                    timer_rows = tokens[:, 1] == self.action_timer_tool_token_id
                    timer_argument_hidden = torch.nn.functional.silu(
                        self.action_timer_argument_mlp_norm(
                            self.action_timer_argument_mlp_in(
                                transformer_out[:, 0]
                            )
                        )
                    )
                    timer_argument_logits = self.action_timer_argument_mlp_out(
                        timer_argument_hidden
                    ).float()
                    logits = torch.where(
                        timer_rows.unsqueeze(-1),
                        logits + timer_argument_logits,
                        logits,
                    )
                if slot == 0 and start_decision_logits is not None:
                    logits = torch.where(
                        (phase_tensor == 0).unsqueeze(-1),
                        start_decision_logits,
                        logits,
                    )
                if logit_masker is not None:
                    logits = logit_masker(
                        slot, phase_tensor, tokens.clone(), logits
                    )
                    if logits.shape != (batch, self.action_card):
                        raise ValueError("action grammar masker changed the logit shape")
                if slot == 0:
                    gate_closed = call_gate_logits < self.action_call_gate_threshold
                    if gate_closed.any():
                        logits = logits.clone()
                        logits[gate_closed, self.action_call_begin_token_id] = -torch.inf
                sampled = sample_token(
                    logits[:, None, None],
                    use_sampling,
                    temperature,
                    min(top_k, self.action_card) if top_k > 0 else top_k,
                )[:, 0, 0]
                if forced_tokens is not None:
                    mask = (
                        forced_mask[:, slot]
                        if forced_mask is not None
                        else torch.ones_like(active)
                    )
                    sampled = torch.where(mask, forced_tokens[:, slot], sampled)
                next_token = torch.where(
                    active,
                    sampled,
                    torch.full_like(sampled, MICRO_PADDING_TOKEN_ID),
                )
                tokens[:, slot] = next_token
                all_logits.append(logits)
                terminal = next_token == MICRO_PADDING_TOKEN_ID
                for terminal_id in self.terminal_action_token_ids:
                    terminal |= next_token == terminal_id
                active &= ~terminal

        return ActionMicroOutput(
            tokens=tokens,
            logits=torch.stack(all_logits, dim=1),
            call_gate_logits=call_gate_logits,
            call_endpoint_logits=call_endpoint_logits,
            call_intent_logits=call_intent_logits,
            call_soft_intent_logits=call_soft_intent_logits,
            audio_summary=self.action_audio_summary(tokens),
            temporal_state=next_temporal_state,
            start_temporal_state=next_start_temporal_state,
            call_gate_temporal_state=next_call_gate_temporal_state,
            call_intent_temporal_state=next_call_intent_temporal_state,
        )

    def forward_micro_audio_depformer(
        self,
        depformer_cb_index: int,
        sequence: torch.Tensor,
        action_summary: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        if action_summary.dim() != 2:
            raise ValueError("same-tick action summary must be [batch, depth_dim]")
        if self.depformer_multi_linear:
            depformer_input = self.depformer_in[depformer_cb_index](transformer_out)
        else:
            depformer_input = self.depformer_in[0](transformer_out)
        if depformer_cb_index == 0:
            previous_token_embedding = self.depformer_text_emb(sequence[:, 0])
        else:
            previous_token_embedding = self.depformer_emb[depformer_cb_index - 1](
                sequence[:, 0]
            )
        depformer_input = (
            depformer_input + previous_token_embedding + action_summary[:, None]
        )
        depformer_out = self.depformer(depformer_input)
        return self.linears[depformer_cb_index](depformer_out)[:, None]

    def forward_micro_audio_training(
        self,
        sequence: torch.Tensor,
        action_targets: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        """Audio logits with per-codebook semantic action timestamp alignment."""

        batch, codebooks, steps = sequence.shape
        if codebooks != self.num_codebooks:
            raise ValueError("unexpected PersonaPlex codebook count")
        summaries = self.action_audio_summary(action_targets)
        depformer_inputs: list[torch.Tensor] = []
        for codebook in range(self.dep_q):
            if self.depformer_multi_linear:
                linear_index = codebook
                if self.depformer_weights_per_step_schedule is not None:
                    linear_index = self.depformer_weights_per_step_schedule[codebook]
                temporal_input = self.depformer_in[linear_index](transformer_out)
            else:
                temporal_input = self.depformer_in[0](transformer_out)
            if codebook == 0:
                previous_embedding = self.depformer_text_emb(sequence[:, 0])
            else:
                previous_embedding = self.depformer_emb[codebook - 1](
                    sequence[:, codebook + self.audio_offset - 1]
                )
            delay = self.delays[self.audio_offset + codebook]
            action_summary = summaries.roll(delay, dims=1)
            if delay:
                action_summary[:, :delay] = 0
            depformer_inputs.append(
                temporal_input + previous_embedding + action_summary
            )
        depformer_input = torch.stack(depformer_inputs, dim=2).view(
            batch * steps, self.dep_q, -1
        )
        depformer_output = self.depformer(depformer_input)
        logits = [
            self.linears[codebook](depformer_output[:, codebook]).view(
                batch, steps, -1
            )
            for codebook in range(self.dep_q)
        ]
        return torch.stack(logits, dim=1)

    def forward_micro_train(
        self,
        codes: torch.Tensor,
        action_tokens: torch.Tensor,
        environment_tokens: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> AgentMicroTrainOutput:
        """Causal training forward for base, action, and input-only environment lanes."""

        if codes.dim() != 3:
            raise ValueError("PersonaPlex codes must be [batch, codebooks, steps]")
        batch, codebooks, steps = codes.shape
        if codebooks != self.num_codebooks or steps < 1:
            raise ValueError("invalid PersonaPlex training-code shape")
        self._validate_micro_tokens(codes, action_tokens, environment_tokens)
        self.validate_action_frame_suffix(action_tokens)
        if action_mask is not None and action_mask.shape != action_tokens.shape:
            raise ValueError("action loss mask must match action microtokens")

        initial_codes = self._get_initial_token().expand(batch, -1, -1)
        delayed_codes = _delay_sequence(self.delays, codes, initial_codes)
        delayed_codes = torch.cat([initial_codes, delayed_codes], dim=2)
        initial_action = torch.full(
            (batch, 1, self.action_slots),
            MICRO_PADDING_TOKEN_ID,
            dtype=torch.long,
            device=codes.device,
        )
        delayed_action = torch.cat([initial_action, action_tokens], dim=1)
        initial_environment = torch.full(
            (batch, 1, self.environment_slots),
            MICRO_PADDING_TOKEN_ID,
            dtype=torch.long,
            device=codes.device,
        )
        delayed_environment = torch.cat(
            [initial_environment, environment_tokens], dim=1
        )

        backbone = self.forward_micro_codes(
            delayed_codes[:, :, :-1],
            delayed_action[:, :-1],
            delayed_environment[:, :-1],
        )
        text_targets = delayed_codes[:, 0, 1:]
        action_logits = self.forward_action_depth_training(
            backbone.transformer_out,
            text_targets,
            action_tokens,
            frame_phase=(
                torch.arange(steps, device=codes.device).view(1, -1).expand(batch, -1)
                % 2
            ),
        )
        audio_logits = self.forward_micro_audio_training(
            delayed_codes[:, :, 1:], action_tokens, backbone.speech_transformer_out
        )
        audio_logits, audio_valid = _undelay_sequence(
            self.delays[self.audio_offset : self.audio_offset + self.dep_q],
            audio_logits,
            fill_value=float("NaN"),
        )
        audio_valid &= (
            codes[:, self.audio_offset : self.audio_offset + self.dep_q]
            != self.zero_token_id
        )
        text_logits, text_valid = _undelay_sequence(
            self.delays[:1], backbone.text_logits, fill_value=float("NaN")
        )
        text_valid &= codes[:, :1] != self.zero_token_id
        response_token_repair_bias = None
        if backbone.response_token_repair_bias is not None:
            response_token_repair_bias, _ = _undelay_sequence(
                self.delays[:1],
                backbone.response_token_repair_bias[:, None],
                fill_value=0.0,
            )
        response_token_repair_hidden = None
        if backbone.response_token_repair_hidden is not None:
            response_token_repair_hidden, _ = _undelay_sequence(
                self.delays[:1],
                backbone.response_token_repair_hidden[:, None],
                fill_value=0.0,
            )
        response_token_repair_source = None
        if backbone.response_token_repair_source is not None:
            response_token_repair_source, _ = _undelay_sequence(
                self.delays[:1],
                backbone.response_token_repair_source[:, None],
                fill_value=0.0,
            )
        response_token_continuation_bias = None
        if backbone.response_token_continuation_bias is not None:
            response_token_continuation_bias, _ = _undelay_sequence(
                self.delays[:1],
                backbone.response_token_continuation_bias[:, None],
                fill_value=0.0,
            )
        response_token_continuation_hidden = None
        if backbone.response_token_continuation_hidden is not None:
            response_token_continuation_hidden, _ = _undelay_sequence(
                self.delays[:1],
                backbone.response_token_continuation_hidden[:, None],
                fill_value=0.0,
            )
        response_token_continuation_source = None
        if backbone.response_token_continuation_source is not None:
            response_token_continuation_source, _ = _undelay_sequence(
                self.delays[:1],
                backbone.response_token_continuation_source[:, None],
                fill_value=0.0,
            )
        if action_mask is None:
            action_mask = action_tokens >= 0
        else:
            action_mask = action_mask.to(dtype=torch.bool) & (action_tokens >= 0)
        return AgentMicroTrainOutput(
            audio_logits=audio_logits,
            audio_mask=audio_valid,
            text_logits=text_logits,
            text_mask=text_valid,
            action_logits=action_logits,
            action_mask=action_mask,
            response_token_repair_bias=response_token_repair_bias,
            response_token_repair_hidden=response_token_repair_hidden,
            response_token_repair_source=response_token_repair_source,
            response_token_continuation_bias=(
                response_token_continuation_bias
            ),
            response_token_continuation_hidden=(
                response_token_continuation_hidden
            ),
            response_token_continuation_source=(
                response_token_continuation_source
            ),
        )

    def forward_micro_gate_train(
        self,
        codes: torch.Tensor,
        action_tokens: torch.Tensor,
        environment_tokens: torch.Tensor,
    ) -> AgentMicroGateTrainOutput:
        """Low-memory raw-code forward for the late adapter and call gate only."""

        if codes.dim() != 3:
            raise ValueError("PersonaPlex codes must be [batch, codebooks, steps]")
        batch, codebooks, steps = codes.shape
        if codebooks != self.num_codebooks or steps < 1:
            raise ValueError("invalid PersonaPlex training-code shape")
        self._validate_micro_tokens(codes, action_tokens, environment_tokens)
        self.validate_action_frame_suffix(action_tokens)

        initial_codes = self._get_initial_token().expand(batch, -1, -1)
        delayed_codes = _delay_sequence(self.delays, codes, initial_codes)
        delayed_codes = torch.cat([initial_codes, delayed_codes], dim=2)
        initial_action = torch.full(
            (batch, 1, self.action_slots),
            MICRO_PADDING_TOKEN_ID,
            dtype=torch.long,
            device=codes.device,
        )
        delayed_action = torch.cat([initial_action, action_tokens], dim=1)
        initial_environment = torch.full(
            (batch, 1, self.environment_slots),
            MICRO_PADDING_TOKEN_ID,
            dtype=torch.long,
            device=codes.device,
        )
        delayed_environment = torch.cat(
            [initial_environment, environment_tokens], dim=1
        )
        backbone = self.forward_micro_codes(
            delayed_codes[:, :, :-1],
            delayed_action[:, :-1],
            delayed_environment[:, :-1],
        )
        text_targets = delayed_codes[:, 0, 1:]
        frame_phase = (
            torch.arange(steps, device=codes.device).view(1, -1).expand(batch, -1)
            % 2
        )
        endpoint_logits, intent_logits = self.forward_call_gate_and_intent_training(
            backbone.transformer_out,
            text_targets,
            frame_phase=frame_phase,
        )
        return AgentMicroGateTrainOutput(
            transformer_out=backbone.transformer_out,
            text_logits=backbone.text_logits,
            call_endpoint_logits=endpoint_logits,
            call_intent_logits=intent_logits,
        )


def agent_micro_causal_loss(
    output: AgentMicroTrainOutput,
    codes: torch.Tensor,
    action_tokens: torch.Tensor,
    *,
    audio_weight: float = 1.0,
    text_weight: float = 1.0,
    action_weight: float = 1.0,
    audio_supervision_mask: Optional[torch.Tensor] = None,
    text_supervision_mask: Optional[torch.Tensor] = None,
    text_target_weights: Optional[torch.Tensor] = None,
    action_target_weights: Optional[torch.Tensor] = None,
) -> AgentLossReport:
    """Compute model-owned losses with optional corpus supervision windows.

    The default masks preserve upstream-compatible all-stream training.  V2
    head warm-up can instead score only assistant audio and aligned assistant
    text after a tool result becomes visible.  Environment tokens never enter
    this signature and can never become prediction targets.
    """

    if min(audio_weight, text_weight, action_weight) < 0:
        raise ValueError("loss weights must be non-negative")
    audio_targets = codes[:, 1 : 1 + output.audio_logits.shape[1]]
    audio_mask = output.audio_mask
    if audio_supervision_mask is not None:
        if audio_supervision_mask.shape != audio_mask.shape:
            raise ValueError("audio supervision mask must match audio targets [B,K,T]")
        audio_mask = audio_mask & audio_supervision_mask.to(
            device=audio_mask.device, dtype=torch.bool
        )
    text_mask = output.text_mask
    if text_supervision_mask is not None:
        if text_supervision_mask.shape != text_mask.shape:
            raise ValueError("text supervision mask must match text targets [B,1,T]")
        text_mask = text_mask & text_supervision_mask.to(
            device=text_mask.device, dtype=torch.bool
        )
    audio_loss, audio_count = _masked_cross_entropy(
        output.audio_logits, audio_targets, audio_mask
    )
    text_targets = codes[:, :1]
    if text_target_weights is None:
        text_loss, text_count = _masked_cross_entropy(
            output.text_logits, text_targets, text_mask
        )
    else:
        if text_target_weights.shape != text_targets.shape:
            raise ValueError("text target weights must match text targets [B,1,T]")
        weights = text_target_weights.to(
            device=output.text_logits.device,
            dtype=torch.float32,
        )
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("text target weights must be finite and non-negative")
        text_count = int(text_mask.sum().item())
        selected_weights = weights[text_mask]
        weight_sum = selected_weights.sum()
        if text_count == 0:
            text_loss = output.text_logits.new_zeros(())
        elif weight_sum <= 0:
            raise ValueError("selected text target weights must have positive mass")
        else:
            per_token = torch.nn.functional.cross_entropy(
                output.text_logits[text_mask].float(),
                text_targets[text_mask],
                reduction="none",
            )
            text_loss = (per_token * selected_weights).sum() / weight_sum
    if action_target_weights is None:
        action_loss, action_count = _masked_cross_entropy(
            output.action_logits, action_tokens, output.action_mask
        )
    else:
        if action_target_weights.shape != action_tokens.shape:
            raise ValueError("action target weights must match action targets [B,T,slots]")
        weights = action_target_weights.to(
            device=output.action_logits.device,
            dtype=torch.float32,
        )
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("action target weights must be finite and non-negative")
        action_mask = output.action_mask.to(dtype=torch.bool)
        action_count = int(action_mask.sum().item())
        selected_weights = weights[action_mask]
        weight_sum = selected_weights.sum()
        if action_count == 0:
            action_loss = output.action_logits.new_zeros(())
        elif weight_sum <= 0:
            raise ValueError("selected action target weights must have positive mass")
        else:
            per_token = torch.nn.functional.cross_entropy(
                output.action_logits[action_mask].float(),
                action_tokens[action_mask],
                reduction="none",
            )
            action_loss = (per_token * selected_weights).sum() / weight_sum
    total = audio_weight * audio_loss + text_weight * text_loss + action_weight * action_loss
    return AgentLossReport(
        total=total,
        audio=audio_loss,
        text=text_loss,
        action=action_loss,
        audio_token_count=audio_count,
        text_token_count=text_count,
        action_token_count=action_count,
    )


@dataclass
class _AgentMicroGenState:
    base_cache: torch.Tensor
    action_cache: torch.Tensor  # [B, action_slots, cache]
    environment_cache: torch.Tensor  # [B, environment_slots, cache]
    base_provided: torch.Tensor
    action_provided: torch.Tensor
    initial: torch.Tensor
    action_temporal_state: torch.Tensor
    action_start_temporal_state: torch.Tensor
    call_gate_temporal_state: torch.Tensor
    call_intent_temporal_state: torch.Tensor
    response_environment_state: torch.Tensor
    response_temporal_state: torch.Tensor
    response_terminal_seen: torch.Tensor
    response_active: torch.Tensor
    response_token_repair_scope_seen: torch.Tensor
    response_token_repair_scope_semantic_seen: torch.Tensor
    speech_pending: torch.Tensor
    speech_pending_terminal_seen: torch.Tensor
    speech_text_eos_complete: torch.Tensor
    speech_result_semantic_count: torch.Tensor
    speech_result_semantic_candidate: torch.Tensor
    graphed_backbone: CUDAGraphed
    graphed_audio: CUDAGraphed
    offset: int = 0
    action_frame_origin: int = 0

    def reset(self) -> None:
        self.offset = 0
        self.base_cache.fill_(-2)
        self.action_cache.fill_(MICRO_PADDING_TOKEN_ID)
        self.environment_cache.fill_(MICRO_PADDING_TOKEN_ID)
        self.base_provided[:] = False
        self.action_provided[:] = False
        self.action_temporal_state.zero_()
        self.action_start_temporal_state.zero_()
        self.call_gate_temporal_state.zero_()
        self.call_intent_temporal_state.zero_()
        self.response_environment_state.zero_()
        self.response_temporal_state.zero_()
        self.response_terminal_seen.zero_()
        self.response_active.zero_()
        self.response_token_repair_scope_seen.zero_()
        self.response_token_repair_scope_semantic_seen.zero_()
        self.speech_pending.zero_()
        self.speech_pending_terminal_seen.zero_()
        self.speech_text_eos_complete.zero_()
        self.speech_result_semantic_count.zero_()
        self.speech_result_semantic_candidate.fill_(-1)
        self.graphed_backbone.reset(warmup_steps=1)
        self.graphed_audio.reset(warmup_steps=1)
        self.action_frame_origin = 0


def _select_first_result_semantic_token(
    sampled_text: torch.Tensor,
    text_logits: torch.Tensor,
    result_scope_seen: torch.Tensor,
    semantic_count: torch.Tensor,
    speech_pending: torch.Tensor,
    semantic_candidate: torch.Tensor,
    *,
    enabled: bool,
) -> torch.Tensor:
    """Arbitrate the first sampled result semantic with a semantic argmax."""

    if not enabled:
        return sampled_text
    if sampled_text.ndim != 1 or text_logits.ndim != 2:
        raise ValueError("result-onset text selection expects [B] and [B,V]")
    if text_logits.shape[0] != sampled_text.shape[0]:
        raise ValueError("result-onset logits and samples must share a batch")
    if text_logits.shape[1] <= 4:
        raise ValueError("result-onset logits contain no semantic vocabulary")
    greedy_text = text_logits[:, 4:].argmax(dim=-1) + 4
    candidate = torch.where(
        semantic_candidate.reshape(-1).ge(4),
        semantic_candidate.reshape(-1),
        greedy_text,
    )
    onset = (
        result_scope_seen.reshape(-1)
        & semantic_count.reshape(-1).eq(0)
        & ~speech_pending.reshape(-1)
        & sampled_text.ge(4)
    )
    if onset.shape != sampled_text.shape:
        raise ValueError("result-onset causal state must contain one value per batch")
    return torch.where(onset, candidate, sampled_text)


@dataclass(frozen=True)
class AgentMicroGeneratedFrame:
    text_token: torch.Tensor
    assistant_audio_tokens: torch.Tensor
    action_tokens: torch.Tensor


class AgentMicroLMGen(StreamingModule[_AgentMicroGenState]):
    """Streaming PersonaPlex generator with a five-slot native action lane."""

    def __init__(
        self,
        lm_model: AgentMicroLMModel,
        *,
        audio_tokens_per_stream: int = 8,
        use_sampling: bool = True,
        use_action_sampling: Optional[bool] = None,
        temp: float = 0.8,
        temp_text: float = 0.7,
        temp_action: float = 0.7,
        top_k: int = 250,
        top_k_text: int = 25,
        top_k_action: int = 25,
        action_logit_masker: Optional[ActionLogitMasker] = None,
        action_frame_observer: Optional[ActionFrameObserver] = None,
        action_decision_observer: Optional[ActionDecisionObserver] = None,
        backbone_frame_observer: Optional[BackboneFrameObserver] = None,
        text_sampling_observer: Optional[TextSamplingObserver] = None,
        result_onset_selection_observer: Optional[
            ResultOnsetSelectionObserver
        ] = None,
        mute_speech_while_tool_pending: bool = False,
        pending_speech_call_begin_token_id: int = 1,
        pending_speech_environment_end_token_id: int = 1,
        pending_speech_terminal_status_ids: Sequence[int] = (3, 4, 5, 6, 7),
        pending_speech_observer: Optional[PendingSpeechObserver] = None,
        stop_speech_on_text_eos: bool = False,
        text_eos_token_id: int = 2,
        stop_speech_on_result_sentence_end: bool = False,
        text_sentence_end_token_ids: Sequence[int] = (),
        result_sentence_min_semantic_tokens: int = 2,
        greedy_first_result_semantic_token: bool = False,
        turn_completion_observer: Optional[TurnCompletionObserver] = None,
        text_prompt_tokens: Optional[Sequence[int]] = None,
        audio_silence_frame_cnt: int = 1,
        cuda_graph_backbone: bool = False,
        cuda_graph_audio: bool = False,
    ) -> None:
        super().__init__()
        if lm_model.training:
            raise ValueError("AgentMicroLMGen requires lm_model.eval()")
        if lm_model.num_audio_codebooks != audio_tokens_per_stream * 2:
            raise ValueError("model audio streams must be assistant plus user codebooks")
        self.lm_model = lm_model
        self.audio_tokens_per_stream = audio_tokens_per_stream
        self.use_sampling = use_sampling
        self.use_action_sampling = (
            use_sampling if use_action_sampling is None else use_action_sampling
        )
        self.temp = temp
        self.temp_text = temp_text
        self.temp_action = temp_action
        self.top_k = top_k
        self.top_k_text = top_k_text
        self.top_k_action = top_k_action
        self.action_logit_masker = action_logit_masker
        self.action_frame_observer = action_frame_observer
        self.action_decision_observer = action_decision_observer
        self.backbone_frame_observer = backbone_frame_observer
        self.text_sampling_observer = text_sampling_observer
        self.result_onset_selection_observer = (
            result_onset_selection_observer
        )
        self.mute_speech_while_tool_pending = bool(
            mute_speech_while_tool_pending
        )
        if not 0 <= pending_speech_call_begin_token_id < lm_model.action_card:
            raise ValueError("pending-speech CALL_BEGIN token is outside action vocabulary")
        if not 0 <= pending_speech_environment_end_token_id < lm_model.environment_card:
            raise ValueError("pending-speech OBS_END token is outside environment vocabulary")
        terminal_status_ids = tuple(
            int(token) for token in pending_speech_terminal_status_ids
        )
        if not terminal_status_ids or len(terminal_status_ids) != len(
            set(terminal_status_ids)
        ):
            raise ValueError("pending-speech terminal status IDs must be unique and non-empty")
        if any(
            token < 0 or token >= lm_model.environment_card
            for token in terminal_status_ids
        ):
            raise ValueError("pending-speech terminal status is outside environment vocabulary")
        from .lm import SILENCE_TOKENS

        if len(SILENCE_TOKENS) < audio_tokens_per_stream:
            raise ValueError("stock silence tokens do not cover assistant audio streams")
        self.pending_speech_call_begin_token_id = int(
            pending_speech_call_begin_token_id
        )
        self.pending_speech_environment_end_token_id = int(
            pending_speech_environment_end_token_id
        )
        self.pending_speech_terminal_status_ids = terminal_status_ids
        self.pending_speech_audio_tokens = tuple(
            int(token) for token in SILENCE_TOKENS[:audio_tokens_per_stream]
        )
        self.pending_speech_observer = pending_speech_observer
        if not 0 <= text_eos_token_id < lm_model.text_linear.weight.shape[0]:
            raise ValueError("text EOS token is outside the text vocabulary")
        self.stop_speech_on_text_eos = bool(stop_speech_on_text_eos)
        self.text_eos_token_id = int(text_eos_token_id)
        sentence_end_ids = tuple(int(token) for token in text_sentence_end_token_ids)
        if len(sentence_end_ids) != len(set(sentence_end_ids)):
            raise ValueError("sentence-end text token IDs must be unique")
        if any(
            token < 4 or token >= lm_model.text_linear.weight.shape[0]
            for token in sentence_end_ids
        ):
            raise ValueError("sentence-end token is outside semantic text vocabulary")
        if stop_speech_on_result_sentence_end and not sentence_end_ids:
            raise ValueError("result-sentence stopping requires punctuation token IDs")
        if result_sentence_min_semantic_tokens < 1:
            raise ValueError("result sentence requires at least one semantic token")
        self.stop_speech_on_result_sentence_end = bool(
            stop_speech_on_result_sentence_end
        )
        self.text_sentence_end_token_ids = sentence_end_ids
        self.result_sentence_min_semantic_tokens = int(
            result_sentence_min_semantic_tokens
        )
        self.greedy_first_result_semantic_token = bool(
            greedy_first_result_semantic_token
        )
        self.turn_completion_observer = turn_completion_observer
        self.max_delay = max(lm_model.delays)
        if audio_silence_frame_cnt < 0:
            raise ValueError("audio_silence_frame_cnt must be non-negative")
        self.text_prompt_tokens = (
            None if text_prompt_tokens is None else tuple(int(token) for token in text_prompt_tokens)
        )
        self.audio_silence_frame_cnt = audio_silence_frame_cnt
        self.cuda_graph_backbone = bool(cuda_graph_backbone)
        self.cuda_graph_audio = bool(cuda_graph_audio)
        self.zero_text_code = 3
        self.voice_prompt: str | None = None
        self.voice_prompt_embeddings: torch.Tensor | None = None
        self.voice_prompt_cache: torch.Tensor | None = None

    def _init_streaming_state(self, batch_size: int) -> _AgentMicroGenState:
        model = self.lm_model
        cache_steps = self.max_delay + 3
        return _AgentMicroGenState(
            base_cache=torch.full(
                (batch_size, model.num_codebooks, cache_steps),
                model.ungenerated_token_id,
                dtype=torch.long,
                device=model.device,
            ),
            action_cache=torch.full(
                (batch_size, model.action_slots, cache_steps),
                MICRO_PADDING_TOKEN_ID,
                dtype=torch.long,
                device=model.device,
            ),
            environment_cache=torch.full(
                (batch_size, model.environment_slots, cache_steps),
                MICRO_PADDING_TOKEN_ID,
                dtype=torch.long,
                device=model.device,
            ),
            base_provided=torch.zeros(
                (batch_size, model.num_codebooks, cache_steps),
                dtype=torch.bool,
                device=model.device,
            ),
            action_provided=torch.zeros(
                (batch_size, model.action_slots, cache_steps),
                dtype=torch.bool,
                device=model.device,
            ),
            initial=model._get_initial_token().expand(batch_size, -1, -1).clone(),
            action_temporal_state=torch.zeros(
                (
                    model.action_temporal_gru.num_layers,
                    batch_size,
                    model.action_depth_dim,
                ),
                dtype=model.action_temporal_in.weight.dtype,
                device=model.device,
            ),
            action_start_temporal_state=torch.zeros(
                (
                    model.action_start_temporal_gru.num_layers,
                    batch_size,
                    model.action_depth_dim,
                ),
                dtype=model.action_start_temporal_in.weight.dtype,
                device=model.device,
            ),
            call_gate_temporal_state=torch.zeros(
                (
                    model.action_call_gate_gru.num_layers,
                    batch_size,
                    model.action_depth_dim,
                ),
                dtype=model.action_call_gate_in.weight.dtype,
                device=model.device,
            ),
            call_intent_temporal_state=torch.zeros(
                (
                    model.action_call_intent_gru.num_layers,
                    batch_size,
                    model.action_depth_dim,
                ),
                dtype=model.action_call_intent_in.weight.dtype,
                device=model.device,
            ),
            response_environment_state=torch.zeros(
                (batch_size, 1, model.dim),
                dtype=model.text_linear.weight.dtype,
                device=model.device,
            ),
            response_temporal_state=torch.zeros(
                (
                    model.response_temporal_gru.num_layers,
                    batch_size,
                    model.response_temporal_dim,
                ),
                dtype=model.response_temporal_base_in.weight.dtype,
                device=model.device,
            ),
            response_terminal_seen=torch.zeros(
                (batch_size, 1, 1), dtype=torch.bool, device=model.device
            ),
            response_active=torch.zeros(
                (batch_size, 1, 1), dtype=torch.bool, device=model.device
            ),
            response_token_repair_scope_seen=torch.zeros(
                (batch_size, 1, 1), dtype=torch.bool, device=model.device
            ),
            response_token_repair_scope_semantic_seen=torch.zeros(
                (batch_size, 1, 1), dtype=torch.bool, device=model.device
            ),
            speech_pending=torch.zeros(
                (batch_size,), dtype=torch.bool, device=model.device
            ),
            speech_pending_terminal_seen=torch.zeros(
                (batch_size,), dtype=torch.bool, device=model.device
            ),
            speech_text_eos_complete=torch.zeros(
                (batch_size,), dtype=torch.bool, device=model.device
            ),
            speech_result_semantic_count=torch.zeros(
                (batch_size,), dtype=torch.long, device=model.device
            ),
            speech_result_semantic_candidate=torch.full(
                (batch_size,), -1, dtype=torch.long, device=model.device
            ),
            graphed_backbone=CUDAGraphed(
                model.forward_micro_codes,
                disable=(
                    model.device.type != "cuda" or not self.cuda_graph_backbone
                ),
            ),
            graphed_audio=CUDAGraphed(
                self._sample_audio,
                disable=model.device.type != "cuda" or not self.cuda_graph_audio,
            ),
        )

    @property
    def next_action_frame(self) -> int:
        """Logical frame sampled by the next normal ``step`` call.

        Prompt frames are deliberately excluded after ``begin_conversation`` so
        the V2 grammar/runtime keeps its stable frame-zero contract.
        """

        state = self._streaming_state
        if state is None:
            raise RuntimeError("wrap generator calls with streaming()")
        return state.offset - 1 - state.action_frame_origin

    def load_voice_prompt_embeddings(self, path: str | Path) -> None:
        """Load a stock PersonaPlex precomputed ``.pt`` voice prompt.

        The file stores base-model input embeddings and the base 17-stream ring
        cache.  It never contains action/environment context, which remains
        padded during replay below.
        """

        prompt_path = Path(path)
        if prompt_path.suffix.lower() != ".pt":
            raise ValueError("AgentMicroLMGen currently accepts precomputed .pt voice prompts")
        if not prompt_path.is_file():
            raise FileNotFoundError(prompt_path)
        try:
            payload = torch.load(prompt_path, map_location="cpu", weights_only=True)
        except TypeError:
            # Compatibility with older PyTorch versions that predate
            # ``weights_only``.  Voice prompts are local project assets.
            payload = torch.load(prompt_path, map_location="cpu")
        if not isinstance(payload, dict) or set(payload) != {"embeddings", "cache"}:
            raise ValueError("voice prompt must contain exactly embeddings and cache tensors")
        embeddings = payload["embeddings"]
        cache = payload["cache"]
        if not isinstance(embeddings, torch.Tensor) or not isinstance(cache, torch.Tensor):
            raise ValueError("voice prompt embeddings and cache must be tensors")
        if embeddings.ndim != 4 or embeddings.shape[2] != 1:
            raise ValueError("voice prompt embeddings must be [frames,batch,1,dim]")
        if embeddings.shape[-1] != self.lm_model.dim:
            raise ValueError("voice prompt embedding dimension disagrees with the model")
        if cache.ndim != 3 or cache.shape[1:] != (
            self.lm_model.num_codebooks,
            self.max_delay + 3,
        ):
            raise ValueError("voice prompt base cache shape disagrees with the model")
        if cache.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("voice prompt base cache must use an integer token dtype")
        if embeddings.shape[1] != cache.shape[0]:
            raise ValueError("voice prompt embedding/cache batch dimensions disagree")
        if embeddings.shape[0] < 1:
            raise ValueError("voice prompt must contain at least one embedding frame")
        parameter = next(self.lm_model.parameters())
        self.voice_prompt = str(prompt_path.resolve())
        self.voice_prompt_embeddings = embeddings.to(
            device=self.lm_model.device,
            dtype=parameter.dtype,
        )
        self.voice_prompt_cache = cache.to(device=self.lm_model.device, dtype=torch.long)

    def _replay_voice_prompt_embeddings(self) -> None:
        """Rebuild transformer KV state from stock base-model prompt embeddings."""

        state = self._streaming_state
        if state is None:
            raise RuntimeError("wrap generator calls with streaming()")
        if state.offset != 0:
            raise RuntimeError("voice prompt replay requires a fresh streaming state")
        embeddings = self.voice_prompt_embeddings
        cache = self.voice_prompt_cache
        if embeddings is None or cache is None:
            return
        batch = state.base_cache.shape[0]
        if embeddings.shape[1] == 1 and cache.shape[0] == 1 and batch > 1:
            embeddings = embeddings.expand(-1, batch, -1, -1)
            cache = cache.expand(batch, -1, -1)
        elif embeddings.shape[1] != batch or cache.shape[0] != batch:
            raise ValueError("voice prompt batch size must match the live generator batch")

        # This mirrors LMGen.step_embeddings: use dummy base forcing only to
        # advance the base ring bookkeeping, then feed the saved base embedding
        # directly through the inherited frozen backbone.  Action/environment
        # embeddings stay PAD=-1, which maps to exact zero.
        initial = state.initial
        assistant_dummy = initial[:, 1 + self.audio_tokens_per_stream :]
        user_dummy = initial[:, 1 : 1 + self.audio_tokens_per_stream]
        text_dummy = torch.full(
            (batch,), self.zero_text_code, dtype=torch.long, device=self.lm_model.device
        )
        for next_embedding in embeddings:
            prepared = self._prepare_step(
                user_dummy,
                assistant_dummy,
                text_dummy,
                torch.full(
                    (batch, self.lm_model.action_slots),
                    MICRO_PADDING_TOKEN_ID,
                    dtype=torch.long,
                    device=self.lm_model.device,
                ),
                torch.zeros(
                    (batch, self.lm_model.action_slots),
                    dtype=torch.bool,
                    device=self.lm_model.device,
                ),
                None,
            )
            while prepared is None:
                prepared = self._prepare_step(
                    user_dummy,
                    assistant_dummy,
                    text_dummy,
                    torch.full(
                        (batch, self.lm_model.action_slots),
                        MICRO_PADDING_TOKEN_ID,
                        dtype=torch.long,
                        device=self.lm_model.device,
                    ),
                    torch.zeros(
                        (batch, self.lm_model.action_slots),
                        dtype=torch.bool,
                        device=self.lm_model.device,
                    ),
                    None,
                )
            _, _, _, _, _ = prepared
            # ``forward_embeddings`` advances exactly the original PersonaPlex
            # temporal transformer state.  Do not use forward_micro_codes here:
            # the stored value is already an embedding, not token IDs.
            self.lm_model.forward_embeddings(next_embedding)
            input_position = (state.offset - 1) % state.base_cache.shape[-1]
            state.base_provided[:, :, input_position] = False
            state.action_provided[:, :, input_position] = False
            state.offset += 1

        # Stored after the stock generator consumed the voice prompt.  Copy it
        # only now so the dummy bookkeeping above cannot overwrite it.
        state.base_cache.copy_(cache)

    def _prompt_step(self, text_token: int) -> None:
        """Feed one stock-style prompt frame without producing an action."""

        from .lm import SILENCE_TOKENS, SINE_TOKENS

        state = self._streaming_state
        if state is None:
            raise RuntimeError("wrap generator calls with streaming()")
        batch = state.base_cache.shape[0]
        device = self.lm_model.device
        if self.audio_tokens_per_stream > len(SILENCE_TOKENS):
            raise ValueError("stock prompt codes do not cover every assistant audio stream")
        assistant_silence = torch.as_tensor(
            SILENCE_TOKENS[: self.audio_tokens_per_stream], dtype=torch.long, device=device
        ).view(1, self.audio_tokens_per_stream, 1).expand(batch, -1, -1)
        user_sine = torch.as_tensor(
            SINE_TOKENS[: self.audio_tokens_per_stream], dtype=torch.long, device=device
        ).view(1, self.audio_tokens_per_stream, 1).expand(batch, -1, -1)
        padding = torch.full(
            (batch, self.lm_model.action_slots),
            MICRO_PADDING_TOKEN_ID,
            dtype=torch.long,
            device=device,
        )
        self.step(
            user_sine,
            assistant_audio_tokens=assistant_silence,
            text_token=torch.full((batch,), text_token, dtype=torch.long, device=device),
            action_tokens=padding,
            action_mask=torch.zeros_like(padding, dtype=torch.bool),
            environment_tokens=None,
            prompt_mode=True,
        )

    @torch.no_grad()
    def step_system_prompts(self) -> int:
        """Apply stock voice/text priming and start logical tool frame zero.

        Precomputed voice prompts are replayed through the original backbone;
        stock silence/text spacer frames then enter through the full agent
        generator with action/environment lanes held at PAD.  The returned
        origin is always zero after ``begin_conversation``.
        """

        state = self._streaming_state
        if state is None:
            raise RuntimeError("wrap generator calls with streaming()")
        if state.offset != 0:
            raise RuntimeError("system prompts require a fresh streaming state")
        self._replay_voice_prompt_embeddings()
        for _ in range(self.audio_silence_frame_cnt):
            self._prompt_step(self.zero_text_code)
        for token in self.text_prompt_tokens or ():
            if token < 0 or token >= self.lm_model.text_card:
                raise ValueError("text prompt token is outside the model text vocabulary")
            self._prompt_step(token)
        for _ in range(self.audio_silence_frame_cnt):
            self._prompt_step(self.zero_text_code)

        return self.begin_conversation()

    @torch.no_grad()
    def begin_conversation(self) -> int:
        """Reset only typed-lane history and make the next sampled frame zero."""

        state = self._streaming_state
        if state is None:
            raise RuntimeError("wrap generator calls with streaming()")
        if state.offset < 1:
            raise RuntimeError("prime the base generator before beginning a conversation")
        origin = state.offset - 1
        state.action_cache.fill_(MICRO_PADDING_TOKEN_ID)
        state.environment_cache.fill_(MICRO_PADDING_TOKEN_ID)
        state.action_provided.zero_()
        state.action_temporal_state.zero_()
        state.action_start_temporal_state.zero_()
        state.call_gate_temporal_state.zero_()
        state.call_intent_temporal_state.zero_()
        state.response_environment_state.zero_()
        state.response_temporal_state.zero_()
        state.response_terminal_seen.zero_()
        state.response_active.zero_()
        state.response_token_repair_scope_seen.zero_()
        state.response_token_repair_scope_semantic_seen.zero_()
        state.speech_pending.zero_()
        state.speech_pending_terminal_seen.zero_()
        state.speech_text_eos_complete.zero_()
        state.speech_result_semantic_count.zero_()
        state.speech_result_semantic_candidate.fill_(-1)
        state.action_frame_origin = origin
        return 0

    def _write_base_slot(self, slot: int, token: torch.Tensor) -> None:
        state = self._streaming_state
        assert state is not None
        position = (state.offset + self.lm_model.delays[slot]) % state.base_cache.shape[-1]
        state.base_cache[:, slot, position] = token
        state.base_provided[:, slot, position] = True

    def _prepare_step(
        self,
        user_tokens: torch.Tensor,
        assistant_audio_tokens: Optional[torch.Tensor],
        text_token: Optional[torch.Tensor],
        action_tokens: Optional[torch.Tensor],
        action_mask: Optional[torch.Tensor],
        environment_tokens: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        state = self._streaming_state
        if state is None:
            raise RuntimeError("wrap generator calls with streaming()")
        model = self.lm_model
        if user_tokens.dim() != 3 or user_tokens.shape[1:] != (
            self.audio_tokens_per_stream,
            1,
        ):
            raise ValueError("user tokens must be [batch, audio_codebooks, one]")
        batch = user_tokens.shape[0]
        if batch != state.base_cache.shape[0]:
            raise ValueError("batch size does not match streaming state")
        if assistant_audio_tokens is not None:
            if assistant_audio_tokens.shape != user_tokens.shape:
                raise ValueError("assistant audio tokens must match user token shape")
            for codebook in range(self.audio_tokens_per_stream):
                self._write_base_slot(
                    1 + codebook, assistant_audio_tokens[:, codebook, 0]
                )
        for codebook in range(self.audio_tokens_per_stream):
            self._write_base_slot(
                1 + self.audio_tokens_per_stream + codebook,
                user_tokens[:, codebook, 0],
            )
        if text_token is not None:
            if text_token.shape != (batch,):
                raise ValueError("text token must be [batch]")
            self._write_base_slot(0, text_token)

        cache_steps = state.base_cache.shape[-1]
        target_position = state.offset % cache_steps
        if action_tokens is not None:
            if action_tokens.shape != (batch, model.action_slots):
                raise ValueError("forced action tokens must be [batch, five]")
            if (action_tokens < MICRO_PADDING_TOKEN_ID).any() or (
                action_tokens >= model.action_card
            ).any():
                raise ValueError("forced action microtoken is outside its vocabulary")
            if action_mask is None:
                action_mask = action_tokens >= 0
            if action_mask.shape != action_tokens.shape:
                raise ValueError("forced action mask must match action tokens")
            action_mask = action_mask.to(dtype=torch.bool)
            sanitized_actions = torch.where(
                action_mask,
                action_tokens,
                torch.full_like(action_tokens, MICRO_PADDING_TOKEN_ID),
            )
            model.validate_action_frame_suffix(sanitized_actions)
            state.action_cache[:, :, target_position] = sanitized_actions
            state.action_provided[:, :, target_position] = action_mask & (
                sanitized_actions >= 0
            )
        if environment_tokens is None:
            state.environment_cache[:, :, target_position] = MICRO_PADDING_TOKEN_ID
        else:
            if environment_tokens.shape != (batch, model.environment_slots):
                raise ValueError("environment tokens must be [batch, five]")
            if (environment_tokens < MICRO_PADDING_TOKEN_ID).any() or (
                environment_tokens >= model.environment_card
            ).any():
                raise ValueError("environment microtoken is outside its vocabulary")
            state.environment_cache[:, :, target_position] = environment_tokens

        for slot, delay in enumerate(model.delays):
            if state.offset <= delay:
                state.base_cache[:, slot, state.offset % cache_steps] = state.initial[:, slot, 0]
                state.base_provided[:, slot, state.offset % cache_steps] = True
        if state.offset == 0:
            state.base_cache[:, :, 0] = state.initial[:, :, 0]
            state.action_cache[:, :, 0] = MICRO_PADDING_TOKEN_ID
            state.environment_cache[:, :, 0] = MICRO_PADDING_TOKEN_ID
            state.offset += 1
            return None

        input_position = (state.offset - 1) % cache_steps
        target_position = state.offset % cache_steps
        return (
            state.base_cache[:, :, input_position : input_position + 1],
            state.action_cache[:, :, input_position].unsqueeze(1),
            state.environment_cache[:, :, input_position].unsqueeze(1),
            state.base_cache[:, :, target_position : target_position + 1],
            state.base_provided[:, :, target_position : target_position + 1],
        )

    def _sample_audio(
        self,
        text_token: torch.Tensor,
        action_frames: torch.Tensor,
        transformer_out: torch.Tensor,
        target_audio_tokens: torch.Tensor,
        audio_provided: torch.Tensor,
    ) -> torch.Tensor:
        model = self.lm_model
        batch = text_token.shape[0]
        if action_frames.shape != (batch, model.dep_q, model.action_slots):
            raise ValueError("action/audio semantic alignment is malformed")
        previous_token = text_token
        sampled: list[torch.Tensor] = []
        assert not model.depformer.is_streaming
        with model.depformer.streaming(batch):
            for codebook in range(model.dep_q):
                action_summary = model.action_audio_summary(
                    action_frames[:, codebook]
                )
                logits = model.forward_micro_audio_depformer(
                    codebook,
                    previous_token[:, None, None],
                    action_summary,
                    transformer_out,
                )
                next_token = sample_token(
                    logits.float(),
                    self.use_sampling,
                    self.temp,
                    min(self.top_k, model.card) if self.top_k > 0 else self.top_k,
                )[:, 0, 0]
                previous_token = torch.where(
                    audio_provided[:, codebook],
                    target_audio_tokens[:, codebook],
                    next_token,
                )
                sampled.append(next_token)
        return torch.stack(sampled, dim=1)

    @torch.no_grad()
    def step(
        self,
        user_tokens: torch.Tensor,
        *,
        assistant_audio_tokens: Optional[torch.Tensor] = None,
        text_token: Optional[torch.Tensor] = None,
        action_tokens: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        environment_tokens: Optional[torch.Tensor] = None,
        prompt_mode: bool = False,
    ) -> AgentMicroGeneratedFrame | None:
        if prompt_mode and (action_tokens is None or action_mask is None):
            raise ValueError("prompt_mode requires explicit all-PAD action tokens and mask")
        prepared = self._prepare_step(
            user_tokens,
            assistant_audio_tokens,
            text_token,
            action_tokens,
            action_mask,
            environment_tokens,
        )
        if prepared is None:
            return None
        base_input, action_input, environment_input, base_target, base_provided = prepared
        state = self._streaming_state
        assert state is not None
        model = self.lm_model
        if self.mute_speech_while_tool_pending and not prompt_mode:
            environment_frame = environment_input[:, 0]
            terminal_status = torch.zeros_like(state.speech_pending)
            for token_id in self.pending_speech_terminal_status_ids:
                terminal_status |= (environment_frame == token_id).any(dim=-1)
            state.speech_pending_terminal_seen |= (
                state.speech_pending & terminal_status
            )
            event_end = (
                environment_frame == self.pending_speech_environment_end_token_id
            ).any(dim=-1)
            completed = (
                state.speech_pending
                & state.speech_pending_terminal_seen
                & event_end
            )
            state.speech_pending &= ~completed
            state.speech_pending_terminal_seen &= ~completed
        backbone = state.graphed_backbone(
            base_input,
            action_input,
            environment_input,
            state.response_environment_state,
            state.response_temporal_state,
            state.response_terminal_seen,
            state.response_active,
            state.response_token_repair_scope_seen,
            state.response_token_repair_scope_semantic_seen,
        )
        state.response_environment_state.copy_(
            backbone.response_environment_state
        )
        state.response_temporal_state.copy_(backbone.response_temporal_state)
        state.response_terminal_seen.copy_(backbone.response_terminal_seen)
        state.response_active.copy_(backbone.response_active)
        state.response_token_repair_scope_seen.copy_(
            backbone.response_token_repair_scope_seen
        )
        state.response_token_repair_scope_semantic_seen.copy_(
            backbone.response_token_repair_scope_semantic_seen
        )
        logical_frame = state.offset - 1 - state.action_frame_origin
        stochastic_text = sample_token(
            backbone.text_logits.float(),
            self.use_sampling,
            self.temp_text,
            (
                min(self.top_k_text, backbone.text_logits.shape[-1])
                if self.top_k_text > 0
                else self.top_k_text
            ),
        )[:, 0, 0]
        current_semantic_argmax = (
            backbone.text_logits[:, 0, 0, 4:].argmax(dim=-1) + 4
        )
        if self.greedy_first_result_semantic_token and not prompt_mode:
            latch_candidate = (
                state.response_token_repair_scope_seen[:, 0, 0]
                & ~state.speech_pending
                & state.speech_result_semantic_count.eq(0)
                & state.speech_result_semantic_candidate.lt(4)
            )
            state.speech_result_semantic_candidate.copy_(
                torch.where(
                    latch_candidate,
                    current_semantic_argmax,
                    state.speech_result_semantic_candidate,
                )
            )
        sampled_text = _select_first_result_semantic_token(
            stochastic_text,
            backbone.text_logits[:, 0, 0],
            state.response_token_repair_scope_seen[:, 0, 0],
            state.speech_result_semantic_count,
            state.speech_pending,
            state.speech_result_semantic_candidate,
            enabled=(
                self.greedy_first_result_semantic_token and not prompt_mode
            ),
        )
        if not prompt_mode and self.result_onset_selection_observer is not None:
            self.result_onset_selection_observer(
                torch.full(
                    (sampled_text.shape[0],),
                    logical_frame,
                    dtype=torch.long,
                    device=sampled_text.device,
                ),
                backbone.text_logits[:, 0, 0].detach(),
                stochastic_text.detach(),
                current_semantic_argmax.detach(),
                state.speech_result_semantic_candidate.detach(),
                sampled_text.detach(),
                state.response_active[:, 0, 0].detach(),
                state.response_token_repair_scope_seen[:, 0, 0].detach(),
                state.speech_pending.detach(),
                state.speech_result_semantic_count.detach(),
            )
        target_position = state.offset % state.base_cache.shape[-1]
        next_text = torch.where(
            base_provided[:, 0, 0], base_target[:, 0, 0], sampled_text
        )
        frame_phase = logical_frame % 2
        if not prompt_mode and self.backbone_frame_observer is not None:
            self.backbone_frame_observer(
                torch.full(
                    (next_text.shape[0],),
                    logical_frame,
                    dtype=torch.long,
                    device=next_text.device,
                ),
                backbone.transformer_out.clone(),
                next_text.clone(),
            )
        if prompt_mode:
            action_frame = state.action_cache[:, :, target_position].clone()
            if (action_frame != MICRO_PADDING_TOKEN_ID).any():
                raise ValueError("prompt_mode action frame must be all PAD")
        else:
            micro = model.sample_action_frame(
                backbone.transformer_out,
                next_text,
                forced_tokens=state.action_cache[:, :, target_position],
                forced_mask=state.action_provided[:, :, target_position],
                temporal_state=state.action_temporal_state,
                start_temporal_state=state.action_start_temporal_state,
                call_gate_temporal_state=state.call_gate_temporal_state,
                call_intent_temporal_state=state.call_intent_temporal_state,
                text_logits=backbone.text_logits[:, 0],
                frame_phase=frame_phase,
                logit_masker=self.action_logit_masker,
                use_sampling=self.use_action_sampling,
                temperature=self.temp_action,
                top_k=self.top_k_action,
            )
            state.action_temporal_state.copy_(micro.temporal_state)
            if micro.start_temporal_state is not None:
                state.action_start_temporal_state.copy_(micro.start_temporal_state)
            state.call_gate_temporal_state.copy_(micro.call_gate_temporal_state)
            state.call_intent_temporal_state.copy_(micro.call_intent_temporal_state)
            action_frame = micro.tokens
            if self.action_decision_observer is not None:
                self.action_decision_observer(
                    torch.full(
                        (action_frame.shape[0],),
                        logical_frame,
                        dtype=torch.long,
                        device=action_frame.device,
                    ),
                    torch.full(
                        (action_frame.shape[0],),
                        frame_phase,
                        dtype=torch.long,
                        device=action_frame.device,
                    ),
                    micro,
                )
            if self.action_frame_observer is not None:
                self.action_frame_observer(
                    torch.full(
                        (action_frame.shape[0],),
                        logical_frame,
                        dtype=torch.long,
                        device=action_frame.device,
                    ),
                    torch.full(
                        (action_frame.shape[0],),
                        frame_phase,
                        dtype=torch.long,
                        device=action_frame.device,
                    ),
                    action_frame.clone(),
                )
        state.action_cache[:, :, target_position] = action_frame

        call_started = torch.zeros(
            (action_frame.shape[0],), dtype=torch.bool, device=model.device
        )
        if not prompt_mode:
            call_started = (
                action_frame == self.pending_speech_call_begin_token_id
            ).any(dim=-1)
        turn_was_complete = state.speech_text_eos_complete.clone()
        if call_started.any():
            state.speech_text_eos_complete &= ~call_started
            state.speech_result_semantic_count.masked_fill_(call_started, 0)
            state.speech_result_semantic_candidate.masked_fill_(
                call_started, -1
            )
            turn_was_complete &= ~call_started
        eos_started = torch.zeros_like(turn_was_complete)
        if self.stop_speech_on_text_eos and not prompt_mode:
            eos_started = (
                state.response_active[:, 0, 0]
                & (next_text == self.text_eos_token_id)
                & ~call_started
            )
            state.speech_text_eos_complete |= eos_started
        sentence_started = torch.zeros_like(turn_was_complete)
        sentence_end = torch.zeros_like(turn_was_complete)
        for token_id in self.text_sentence_end_token_ids:
            sentence_end |= next_text == token_id
        semantic_token = (next_text >= 4) & ~sentence_end
        if (
            self.stop_speech_on_result_sentence_end
            or self.greedy_first_result_semantic_token
        ) and not prompt_mode:
            semantic_count_active = state.response_active[:, 0, 0]
            if self.greedy_first_result_semantic_token:
                semantic_count_active = semantic_count_active | (
                    state.response_token_repair_scope_seen[:, 0, 0]
                )
            state.speech_result_semantic_count += (
                semantic_count_active
                & semantic_token
                & ~state.speech_pending
                & ~call_started
                & ~turn_was_complete
            ).to(dtype=torch.long)
        if self.stop_speech_on_result_sentence_end and not prompt_mode:
            sentence_started = (
                state.response_active[:, 0, 0]
                & sentence_end
                & (
                    state.speech_result_semantic_count
                    >= self.result_sentence_min_semantic_tokens
                )
                & ~call_started
                & ~turn_was_complete
            )
            state.speech_text_eos_complete |= sentence_started
        if not prompt_mode and self.turn_completion_observer is not None:
            self.turn_completion_observer(
                torch.full(
                    (next_text.shape[0],),
                    logical_frame,
                    dtype=torch.long,
                    device=next_text.device,
                ),
                eos_started.clone(),
                sentence_started.clone(),
            )

        mute_speech = turn_was_complete
        if self.mute_speech_while_tool_pending and not prompt_mode:
            state.speech_pending |= call_started
            state.speech_pending_terminal_seen &= ~call_started
            pending_mute = state.speech_pending.clone()
            mute_speech |= pending_mute
            if self.pending_speech_observer is not None:
                self.pending_speech_observer(
                    torch.full(
                        (action_frame.shape[0],),
                        logical_frame,
                        dtype=torch.long,
                        device=model.device,
                    ),
                    pending_mute,
                )

        next_text = torch.where(
            mute_speech,
            torch.full_like(next_text, self.zero_text_code),
            next_text,
        )

        # The batched repair policy can infer response onset from teacher-forced
        # causal inputs. During native generation the authoritative event is the
        # token that was actually emitted above. Latch it immediately so an
        # onset-only scalar cannot survive into another sampling step even when
        # delayed cache/output alignment postpones observation of that token.
        if model.response_token_repair_onset_generated_semantic_latch:
            emitted_scoped_semantic = (
                (next_text >= 4)
                & state.response_active[:, 0, 0]
                & state.response_token_repair_scope_seen[:, 0, 0]
            )
            state.response_token_repair_scope_semantic_seen[:, 0, 0] |= (
                emitted_scoped_semantic
            )

        if not prompt_mode and self.text_sampling_observer is not None:
            self.text_sampling_observer(
                torch.full(
                    (next_text.shape[0],),
                    logical_frame,
                    dtype=torch.long,
                    device=next_text.device,
                ),
                backbone.text_logits[:, 0, 0].detach(),
                sampled_text.detach(),
                next_text.detach(),
                base_provided[:, 0, 0].detach(),
            )

        cache_steps = state.action_cache.shape[-1]
        semantic_positions = torch.tensor(
            [
                (target_position - model.delays[model.audio_offset + codebook])
                % cache_steps
                for codebook in range(model.dep_q)
            ],
            dtype=torch.long,
            device=model.device,
        ).view(1, 1, -1).expand(action_frame.shape[0], model.action_slots, -1)
        audio_action_frames = state.action_cache.gather(2, semantic_positions).permute(
            0, 2, 1
        )
        sampled_audio = state.graphed_audio(
            next_text,
            audio_action_frames,
            backbone.speech_transformer_out,
            base_target[:, 1 : model.dep_q + 1, 0],
            base_provided[:, 1 : model.dep_q + 1, 0],
        )
        next_audio = torch.where(
            base_provided[:, 1 : model.dep_q + 1, 0],
            base_target[:, 1 : model.dep_q + 1, 0],
            sampled_audio,
        )
        mute_audio = mute_speech | eos_started
        if mute_audio.any():
            assistant_silence = torch.as_tensor(
                self.pending_speech_audio_tokens,
                dtype=next_audio.dtype,
                device=next_audio.device,
            ).view(1, -1)
            next_audio[:, : self.audio_tokens_per_stream] = torch.where(
                mute_audio[:, None],
                assistant_silence.expand(next_audio.shape[0], -1),
                next_audio[:, : self.audio_tokens_per_stream],
            )
        state.base_cache[:, 0, target_position] = next_text
        state.base_cache[:, 1 : model.dep_q + 1, target_position] = next_audio
        input_position = (state.offset - 1) % cache_steps
        state.base_provided[:, :, input_position] = False
        state.action_provided[:, :, input_position] = False

        if state.offset <= self.max_delay:
            state.offset += 1
            return None
        batch = state.base_cache.shape[0]
        output_delays = torch.tensor(
            model.delays[: 1 + self.audio_tokens_per_stream],
            dtype=torch.long,
            device=model.device,
        )
        output_index = (
            (state.offset - self.max_delay + output_delays) % cache_steps
        ).view(1, -1, 1).expand(batch, -1, 1)
        gathered = state.base_cache[:, : 1 + self.audio_tokens_per_stream].gather(
            2, output_index
        )
        action_position = (state.offset - self.max_delay) % cache_steps
        frame = AgentMicroGeneratedFrame(
            text_token=gathered[:, 0, 0].clone(),
            assistant_audio_tokens=gathered[:, 1:, 0].clone(),
            action_tokens=state.action_cache[:, :, action_position].clone(),
        )
        state.offset += 1
        return frame
