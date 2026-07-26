"""Native action/result-lane extension for PersonaPlex-Agent.

This module deliberately leaves ``lm.py`` untouched.  ``AgentLMModel`` reuses
the pretrained PersonaPlex temporal transformer and audio heads while adding:

* a generated ``model_action`` vocabulary/head;
* an input-only ``environment_result`` vocabulary;
* an action projection reserved for same-tick speech/action conditioning.

The companion AgentLMGen/cache implementation will own the two new causal
slots.  Keeping this layer separate lets a base PersonaPlex checkpoint load
with the new weights initialized explicitly and trained in stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..modules.streaming import StreamingModule
from ..utils.sampling import sample_token
from .lm import LMModel, _delay_sequence, _undelay_sequence


@dataclass(frozen=True)
class AgentLMOutput:
    """Temporal-backbone output with independent text and action logits."""

    transformer_out: torch.Tensor
    text_logits: torch.Tensor
    action_logits: torch.Tensor


@dataclass(frozen=True)
class AgentLMTrainOutput:
    """Causally aligned training logits for the native typed lanes.

    ``environment_result`` deliberately has no logits or mask: it is an
    executor-authored input, never a model prediction target.
    """

    audio_logits: torch.Tensor
    audio_mask: torch.Tensor
    text_logits: torch.Tensor
    text_mask: torch.Tensor
    action_logits: torch.Tensor
    action_mask: torch.Tensor


@dataclass(frozen=True)
class AgentLossReport:
    """Losses for trainable model outputs only (audio, text, and action)."""

    total: torch.Tensor
    audio: torch.Tensor
    text: torch.Tensor
    action: torch.Tensor
    audio_token_count: int
    text_token_count: int
    action_token_count: int
    environment_token_count: int = 0


class AgentLMModel(LMModel):
    """PersonaPlex temporal model plus explicit typed agent lanes.

    ``sequence`` remains the original 17-stream PersonaPlex cache:
    assistant text, 8 assistant-audio codebooks, and 8 user-audio codebooks.
    Action and environment tokens are passed separately until AgentLMGen
    introduces the permanent 19-stream ring buffer.
    """

    def __init__(
        self,
        *args,
        action_card: int = 256,
        environment_card: int = 256,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if action_card < 2 or environment_card < 2:
            raise ValueError("typed lane vocabularies must reserve at least a token plus padding")
        self.action_card = action_card
        self.environment_card = environment_card
        self.action_emb = self.EmbeddingFactory(action_card + 1, self.dim)
        self.environment_emb = self.EmbeddingFactory(environment_card + 1, self.dim)

        # ``EmbeddingFactory`` inherits the base model device/dtype.  Plain
        # ``torch.nn.Linear`` does not, so use the pretrained text head as the
        # source of truth.  Without this, direct CUDA construction produces a
        # CPU action head and fails on the first real checkpoint forward pass.
        lane_factory_kwargs = {
            "device": self.text_linear.weight.device,
            "dtype": self.text_linear.weight.dtype,
        }

        # Only model action is predicted. Environment results are trusted
        # causal observations supplied by the executor, never model targets.
        self.action_linear = torch.nn.Linear(
            self.dim,
            action_card + 1,
            bias=self.text_linear.bias is not None,
            **lane_factory_kwargs,
        )

        depformer_dim = self.depformer_in[0].out_features
        self.action_to_depformer = torch.nn.Linear(
            self.dim,
            depformer_dim,
            bias=False,
            **lane_factory_kwargs,
        )

    @property
    def action_padding_token_id(self) -> int:
        return self.action_card

    @property
    def no_action_token_id(self) -> int:
        """The trainable categorical no-op action; padding is never sampled."""

        return 0

    @property
    def environment_padding_token_id(self) -> int:
        return self.environment_card

    def _validate_typed_tokens(
        self,
        sequence: torch.Tensor,
        action_tokens: torch.Tensor,
        environment_tokens: torch.Tensor,
    ) -> None:
        if sequence.dim() != 3:
            raise ValueError("PersonaPlex sequence must be [batch, streams, steps]")
        batch, _, steps = sequence.shape
        expected = (batch, steps)
        if action_tokens.shape != expected:
            raise ValueError(f"action tokens must have shape {expected}, got {tuple(action_tokens.shape)}")
        if environment_tokens.shape != expected:
            raise ValueError(
                f"environment tokens must have shape {expected}, got {tuple(environment_tokens.shape)}"
            )
        if (action_tokens > self.action_card).any() or (action_tokens < self.zero_token_id).any():
            raise ValueError("action token is outside its vocabulary")
        if (environment_tokens > self.environment_card).any() or (environment_tokens < self.zero_token_id).any():
            raise ValueError("environment token is outside its vocabulary")

    def embed_agent_codes(
        self,
        sequence: torch.Tensor,
        action_tokens: torch.Tensor,
        environment_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Sum all five semantic streams before the shared temporal backbone."""

        self._validate_typed_tokens(sequence, action_tokens, environment_tokens)
        return (
            self.embed_codes(sequence)
            + self.action_emb(action_tokens)
            + self.environment_emb(environment_tokens)
        )

    def forward_agent_codes(
        self,
        sequence: torch.Tensor,
        action_tokens: torch.Tensor,
        environment_tokens: torch.Tensor,
    ) -> AgentLMOutput:
        """Run the shared backbone and expose independent action logits."""

        embeddings = self.embed_agent_codes(sequence, action_tokens, environment_tokens)
        transformer_out = self.transformer(embeddings)
        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)
        assert isinstance(transformer_out, torch.Tensor)
        return AgentLMOutput(
            transformer_out=transformer_out,
            text_logits=self.text_linear(transformer_out)[:, None],
            action_logits=self.action_linear(transformer_out)[:, None],
        )

    def action_depformer_bias(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """Projection consumed by the future AgentLMGen audio-depth step.

        This is what will make a same-tick action (for example, a tool call)
        influence the assistant audio decoder instead of merely appearing in
        temporal context on a later frame.
        """

        if action_tokens.dim() != 2:
            raise ValueError("action tokens must be [batch, steps]")
        return self.action_to_depformer(self.action_emb(action_tokens))

    def forward_agent_depformer(
        self,
        depformer_cb_index: int,
        sequence: torch.Tensor,
        action_token: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        """Audio-codebook logits conditioned on the same-tick action token."""

        if action_token.dim() != 1:
            raise ValueError("same-tick action token must be [batch]")
        depformer_input = transformer_out
        if self.depformer_multi_linear:
            depformer_input = self.depformer_in[depformer_cb_index](depformer_input)
        else:
            depformer_input = self.depformer_in[0](depformer_input)
        if depformer_cb_index == 0:
            previous_token_embedding = self.depformer_text_emb(sequence[:, 0])
        else:
            previous_token_embedding = self.depformer_emb[depformer_cb_index - 1](sequence[:, 0])
        action_bias = self.action_to_depformer(self.action_emb(action_token))[:, None]
        depformer_input = depformer_input + previous_token_embedding + action_bias
        depformer_out = self.depformer(depformer_input)
        logits = self.linears[depformer_cb_index](depformer_out)
        return logits[:, None]

    def forward_agent_depformer_training(
        self,
        sequence: torch.Tensor,
        action_tokens: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced audio logits conditioned on the aligned action lane.

        At inference, :class:`AgentLMGen` samples an action before its audio
        depth decoder step.  Training uses the same-tick ground-truth action
        token, which is the standard teacher-forcing counterpart of that
        causal generation order.
        """

        batch, codebooks, steps = sequence.shape
        if codebooks != self.num_codebooks:
            raise ValueError("unexpected PersonaPlex codebook count")
        if action_tokens.shape != (batch, steps):
            raise ValueError("action tokens must align with the audio target steps")
        if transformer_out.shape[:2] != (batch, steps):
            raise ValueError("transformer output must align with the audio target steps")

        depformer_inputs: list[torch.Tensor] = []
        for codebook in range(self.dep_q):
            if self.depformer_multi_linear:
                linear_index = codebook
                if self.depformer_weights_per_step_schedule is not None:
                    linear_index = self.depformer_weights_per_step_schedule[codebook]
                transformer_in = self.depformer_in[linear_index](transformer_out)
            else:
                transformer_in = self.depformer_in[0](transformer_out)
            if codebook == 0:
                previous_token_embedding = self.depformer_text_emb(sequence[:, 0])
            else:
                previous_token_embedding = self.depformer_emb[codebook - 1](
                    sequence[:, codebook + self.audio_offset - 1]
                )
            delay = self.delays[self.audio_offset + codebook]
            aligned_actions = action_tokens.roll(delay, dims=1)
            if delay:
                aligned_actions[:, :delay] = self.action_padding_token_id
            action_bias = self.action_depformer_bias(aligned_actions)
            depformer_inputs.append(transformer_in + previous_token_embedding + action_bias)

        depformer_input = torch.stack(depformer_inputs, dim=2)
        depformer_output = self.depformer(depformer_input.view(batch * steps, self.dep_q, -1))
        all_logits: list[torch.Tensor] = []
        for codebook in range(self.dep_q):
            logits = self.linears[codebook](depformer_output[:, codebook])
            all_logits.append(logits.view(batch, steps, -1))
        return torch.stack(all_logits, dim=1)

    def forward_agent_train(
        self,
        codes: torch.Tensor,
        action_tokens: torch.Tensor,
        environment_tokens: torch.Tensor,
    ) -> AgentLMTrainOutput:
        """Build the delayed causal training graph for all typed lanes.

        ``action_tokens`` and ``environment_tokens`` are frame-aligned
        ``[batch, steps]`` inputs.  Both are delayed by one model step before
        entering the temporal transformer.  Action receives a prediction loss;
        environment results only condition later predictions and have no head
        or loss path.
        """

        if codes.dim() != 3:
            raise ValueError("PersonaPlex codes must be [batch, codebooks, steps]")
        batch, codebooks, steps = codes.shape
        if codebooks != self.num_codebooks:
            raise ValueError("unexpected PersonaPlex codebook count")
        if steps < 1:
            raise ValueError("at least one frame is required for causal training")
        self._validate_typed_tokens(codes, action_tokens, environment_tokens)

        initial_codes = self._get_initial_token().expand(batch, -1, -1)
        delayed_codes = _delay_sequence(self.delays, codes, initial_codes)
        delayed_codes = torch.cat([initial_codes, delayed_codes], dim=2)

        initial_action = torch.full(
            (batch, 1, 1),
            self.action_padding_token_id,
            dtype=torch.long,
            device=codes.device,
        )
        delayed_action = _delay_sequence([0], action_tokens[:, None], initial_action)
        delayed_action = torch.cat([initial_action, delayed_action], dim=2)

        initial_environment = torch.full(
            (batch, 1, 1),
            self.zero_token_id,
            dtype=torch.long,
            device=codes.device,
        )
        delayed_environment = _delay_sequence(
            [0], environment_tokens[:, None], initial_environment
        )
        delayed_environment = torch.cat([initial_environment, delayed_environment], dim=2)

        output = self.forward_agent_codes(
            delayed_codes[:, :, :-1],
            delayed_action[:, 0, :-1],
            delayed_environment[:, 0, :-1],
        )
        audio_logits = self.forward_agent_depformer_training(
            delayed_codes[:, :, 1:],
            delayed_action[:, 0, 1:],
            output.transformer_out,
        )

        audio_logits, audio_mask = _undelay_sequence(
            self.delays[self.audio_offset : self.audio_offset + self.dep_q],
            audio_logits,
            fill_value=float("NaN"),
        )
        audio_mask &= (
            codes[:, self.audio_offset : self.audio_offset + self.dep_q]
            != self.zero_token_id
        )
        text_logits, text_mask = _undelay_sequence(
            self.delays[:1], output.text_logits, fill_value=float("NaN")
        )
        text_mask &= codes[:, :1] != self.zero_token_id

        # The reserved action-padding ID is valid as context but never a
        # target.  A real no-op is the explicit token ID 0.
        action_mask = (
            (action_tokens >= self.no_action_token_id)
            & (action_tokens < self.action_card)
        )[:, None]
        return AgentLMTrainOutput(
            audio_logits=audio_logits,
            audio_mask=audio_mask,
            text_logits=text_logits,
            text_mask=text_mask,
            action_logits=output.action_logits,
            action_mask=action_mask,
        )


def _masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Mean CE over explicit valid targets, with no implicit pad semantics."""

    if logits.shape[:-1] != targets.shape or mask.shape != targets.shape:
        raise ValueError("logits, targets, and mask must have matching frame dimensions")
    mask = mask.to(dtype=torch.bool)
    token_count = int(mask.sum().item())
    if token_count == 0:
        return logits.new_zeros(()), 0
    # Keep loss accumulation in fp32 even when the checkpoint/lane adapters
    # live in bf16 during local multi-GPU experimentation.
    selected_logits = logits[mask].float()
    selected_targets = targets[mask]
    return torch.nn.functional.cross_entropy(selected_logits, selected_targets), token_count


def agent_causal_loss(
    output: AgentLMTrainOutput,
    codes: torch.Tensor,
    action_tokens: torch.Tensor,
    *,
    audio_weight: float = 1.0,
    text_weight: float = 1.0,
    action_weight: float = 1.0,
) -> AgentLossReport:
    """Compute losses for model-owned lanes only.

    Tool outcomes are intentionally absent from this signature.  They shape
    future predictions through ``forward_agent_train`` but can never become a
    generated target or receive a cross-entropy term.
    """

    if min(audio_weight, text_weight, action_weight) < 0:
        raise ValueError("loss weights must be non-negative")
    if codes.dim() != 3 or action_tokens.dim() != 2:
        raise ValueError("codes must be [batch, codebooks, steps] and actions [batch, steps]")
    if action_tokens.shape != (codes.shape[0], codes.shape[2]):
        raise ValueError("action target shape must match the base frame timeline")

    audio_targets = codes[:, 1 : 1 + output.audio_logits.shape[1]]
    text_targets = codes[:, :1]
    action_targets = action_tokens[:, None]
    audio_loss, audio_count = _masked_cross_entropy(
        output.audio_logits, audio_targets, output.audio_mask
    )
    text_loss, text_count = _masked_cross_entropy(
        output.text_logits, text_targets, output.text_mask
    )
    action_loss, action_count = _masked_cross_entropy(
        output.action_logits, action_targets, output.action_mask
    )
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
class _AgentLMGenState:
    base_cache: torch.Tensor
    action_cache: torch.Tensor
    environment_cache: torch.Tensor
    base_provided: torch.Tensor
    action_provided: torch.Tensor
    initial: torch.Tensor
    offset: int = 0

    def reset(self) -> None:
        self.offset = 0
        self.base_provided[:] = False
        self.action_provided[:] = False


@dataclass(frozen=True)
class AgentGeneratedFrame:
    """One generated frame: speech text/audio plus a model-generated action."""

    text_token: torch.Tensor
    audio_tokens: torch.Tensor  # Assistant audio codebooks only.
    action_token: torch.Tensor


class AgentLMGen(StreamingModule[_AgentLMGenState]):
    """Minimal native typed-lane generator for the first action-lane prototype.

    The original PersonaPlex cache has 17 streams. This generator keeps its
    audio/text cache separate from the new action and environment caches while
    feeding all five semantic streams through :class:`AgentLMModel` each tick.
    It intentionally supports one action token and one environment token per
    80 ms frame; the later action micro-decoder expands that to the 10-token
    160 ms packet defined by the training schema.
    """

    def __init__(
        self,
        lm_model: AgentLMModel,
        *,
        audio_tokens_per_stream: int = 8,
        use_sampling: bool = True,
        temp: float = 0.8,
        temp_text: float = 0.7,
        top_k: int = 250,
        top_k_text: int = 25,
    ) -> None:
        super().__init__()
        if lm_model.training:
            raise ValueError("AgentLMGen requires lm_model.eval()")
        if lm_model.num_audio_codebooks != audio_tokens_per_stream * 2:
            raise ValueError("model audio streams must be agent plus user codebooks")
        self.lm_model = lm_model
        self.audio_tokens_per_stream = audio_tokens_per_stream
        self.use_sampling = use_sampling
        self.temp = temp
        self.temp_text = temp_text
        self.top_k = top_k
        self.top_k_text = top_k_text
        self.max_delay = max(lm_model.delays)

    def _init_streaming_state(self, batch_size: int) -> _AgentLMGenState:
        lm_model = self.lm_model
        cache_steps = self.max_delay + 3
        return _AgentLMGenState(
            base_cache=torch.full(
                (batch_size, lm_model.num_codebooks, cache_steps),
                lm_model.ungenerated_token_id,
                dtype=torch.long,
                device=lm_model.device,
            ),
            action_cache=torch.full(
                (batch_size, cache_steps),
                lm_model.ungenerated_token_id,
                dtype=torch.long,
                device=lm_model.device,
            ),
            environment_cache=torch.full(
                (batch_size, cache_steps),
                lm_model.zero_token_id,
                dtype=torch.long,
                device=lm_model.device,
            ),
            base_provided=torch.zeros(
                (batch_size, lm_model.num_codebooks, cache_steps),
                dtype=torch.bool,
                device=lm_model.device,
            ),
            action_provided=torch.zeros(
                (batch_size, cache_steps),
                dtype=torch.bool,
                device=lm_model.device,
            ),
            initial=lm_model._get_initial_token().expand(batch_size, -1, -1).clone(),
        )

    def _write_base_slot(self, slot: int, token: torch.Tensor) -> None:
        state = self._streaming_state
        assert state is not None
        delay = self.lm_model.delays[slot]
        position = (state.offset + delay) % state.base_cache.shape[-1]
        state.base_cache[:, slot, position] = token
        state.base_provided[:, slot, position] = True

    def _prepare_step(
        self,
        user_tokens: torch.Tensor,
        assistant_audio_tokens: Optional[torch.Tensor],
        text_token: Optional[torch.Tensor],
        action_token: Optional[torch.Tensor],
        environment_token: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        state = self._streaming_state
        if state is None:
            raise RuntimeError("wrap AgentLMGen calls with streaming_forever() or streaming()")
        lm_model = self.lm_model
        if user_tokens.dim() != 3 or user_tokens.shape[1:] != (self.audio_tokens_per_stream, 1):
            raise ValueError("user tokens must be [batch, audio_tokens_per_stream, 1]")
        batch = user_tokens.shape[0]
        if batch != state.base_cache.shape[0]:
            raise ValueError("batch size does not match streaming state")
        if assistant_audio_tokens is not None:
            if assistant_audio_tokens.shape != user_tokens.shape:
                raise ValueError("assistant audio tokens must match user token shape")
            for codebook in range(self.audio_tokens_per_stream):
                self._write_base_slot(1 + codebook, assistant_audio_tokens[:, codebook, 0])
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
        if action_token is not None:
            if action_token.shape != (batch,):
                raise ValueError("action token must be [batch]")
            if (action_token < lm_model.no_action_token_id).any() or (
                action_token >= lm_model.action_card
            ).any():
                raise ValueError("forced action token must be a non-padding action ID")
            state.action_cache[:, target_position] = action_token
            state.action_provided[:, target_position] = True
        # Results are trusted observations. An absent result is an explicit
        # zero embedding, not a model-generated padding/result token.
        if environment_token is None:
            state.environment_cache[:, target_position] = lm_model.zero_token_id
        else:
            if environment_token.shape != (batch,):
                raise ValueError("environment token must be [batch]")
            if (environment_token < lm_model.zero_token_id).any() or (
                environment_token > lm_model.environment_card
            ).any():
                raise ValueError("environment token is outside its vocabulary")
            state.environment_cache[:, target_position] = environment_token

        for slot, delay in enumerate(lm_model.delays):
            if state.offset <= delay:
                state.base_cache[:, slot, state.offset % cache_steps] = state.initial[:, slot, 0]
                state.base_provided[:, slot, state.offset % cache_steps] = True
        if state.offset == 0:
            state.base_cache[:, :, 0] = state.initial[:, :, 0]
            state.action_cache[:, 0] = lm_model.action_padding_token_id
            state.environment_cache[:, 0] = lm_model.zero_token_id
            state.offset += 1
            return None

        input_position = (state.offset - 1) % cache_steps
        target_position = state.offset % cache_steps
        return (
            state.base_cache[:, :, input_position : input_position + 1],
            state.action_cache[:, input_position : input_position + 1],
            state.environment_cache[:, input_position : input_position + 1],
            state.base_cache[:, :, target_position : target_position + 1],
            state.base_provided[:, :, target_position : target_position + 1],
        )

    def _sample_audio(
        self,
        text_token: torch.Tensor,
        action_tokens: torch.Tensor,
        transformer_out: torch.Tensor,
        target_audio_tokens: torch.Tensor,
        audio_provided: torch.Tensor,
    ) -> torch.Tensor:
        lm_model = self.lm_model
        batch = text_token.shape[0]
        if action_tokens.shape != (batch, lm_model.dep_q):
            raise ValueError("audio conditioning actions must align with every depth codebook")
        previous_token = text_token
        sampled: list[torch.Tensor] = []
        assert not lm_model.depformer.is_streaming
        with lm_model.depformer.streaming(batch):
            for codebook in range(lm_model.dep_q):
                logits = lm_model.forward_agent_depformer(
                    codebook,
                    previous_token[:, None, None],
                    action_tokens[:, codebook],
                    transformer_out,
                )
                next_token = sample_token(logits.float(), self.use_sampling, self.temp, self.top_k)[:, 0, 0]
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
        action_token: Optional[torch.Tensor] = None,
        environment_token: Optional[torch.Tensor] = None,
    ) -> AgentGeneratedFrame | None:
        prepared = self._prepare_step(
            user_tokens,
            assistant_audio_tokens,
            text_token,
            action_token,
            environment_token,
        )
        if prepared is None:
            return None
        base_input, action_input, environment_input, base_target, base_provided = prepared
        state = self._streaming_state
        assert state is not None
        lm_model = self.lm_model

        output = lm_model.forward_agent_codes(base_input, action_input, environment_input)
        sampled_text = sample_token(output.text_logits.float(), self.use_sampling, self.temp_text, self.top_k_text)[:, 0, 0]
        # The final action logit is reserved padding/BOS context.  It must
        # never be emitted as an executor-visible model action.
        sampled_action = sample_token(
            output.action_logits[..., :lm_model.action_card].float(),
            self.use_sampling,
            self.temp_text,
            self.top_k_text,
        )[:, 0, 0]
        target_position = state.offset % state.base_cache.shape[-1]
        next_text = torch.where(base_provided[:, 0, 0], base_target[:, 0, 0], sampled_text)
        next_action = torch.where(
            state.action_provided[:, target_position],
            state.action_cache[:, target_position],
            sampled_action,
        )
        # Store the generated action before audio depth decoding.  A residual
        # audio codebook with delay d must use the action belonging to its
        # semantic frame (target_position - d), not blindly the newest action.
        state.action_cache[:, target_position] = next_action
        cache_steps = state.action_cache.shape[-1]
        action_positions = torch.tensor(
            [
                (target_position - lm_model.delays[lm_model.audio_offset + codebook])
                % cache_steps
                for codebook in range(lm_model.dep_q)
            ],
            dtype=torch.long,
            device=lm_model.device,
        ).view(1, -1).expand(next_action.shape[0], -1)
        audio_actions = state.action_cache.gather(1, action_positions)
        sampled_audio = self._sample_audio(
            next_text,
            audio_actions,
            output.transformer_out,
            base_target[:, 1 : lm_model.dep_q + 1, 0],
            base_provided[:, 1 : lm_model.dep_q + 1, 0],
        )

        state.base_cache[:, 0, target_position] = next_text
        state.base_cache[:, 1 : lm_model.dep_q + 1, target_position] = torch.where(
            base_provided[:, 1 : lm_model.dep_q + 1, 0],
            base_target[:, 1 : lm_model.dep_q + 1, 0],
            sampled_audio,
        )
        input_position = (state.offset - 1) % state.base_cache.shape[-1]
        state.base_provided[:, :, input_position] = False
        state.action_provided[:, input_position] = False

        if state.offset <= self.max_delay:
            state.offset += 1
            return None
        batch = state.base_cache.shape[0]
        cache_steps = state.base_cache.shape[-1]
        output_delays = torch.tensor(
            lm_model.delays[: 1 + self.audio_tokens_per_stream],
            dtype=torch.long,
            device=lm_model.device,
        )
        output_index = (
            (state.offset - self.max_delay + output_delays) % cache_steps
        ).view(1, -1, 1).expand(batch, -1, 1)
        gathered = state.base_cache[:, : 1 + self.audio_tokens_per_stream].gather(
            dim=2, index=output_index
        )
        action_output_position = (state.offset - self.max_delay) % cache_steps
        frame = AgentGeneratedFrame(
            text_token=gathered[:, 0, 0].clone(),
            audio_tokens=gathered[:, 1:, 0].clone(),
            action_token=state.action_cache[:, action_output_position].clone(),
        )
        state.offset += 1
        return frame
