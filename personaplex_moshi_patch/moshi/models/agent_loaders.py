"""Checkpoint loader for the PersonaPlex-Agent typed-lane model.

The published PersonaPlex checkpoint contains only the pretrained 17-stream
audio/text model.  This loader attaches it to :class:`AgentLMModel`, preserves
the upstream compatibility patches, and initializes only the newly introduced
action/result-lane weights.  It is intentionally separate from ``loaders.py``
so stock PersonaPlex loading remains untouched.
"""

from __future__ import annotations

import math
from pathlib import Path

from safetensors.torch import load_file
import torch

from .agent_lm import AgentLMModel
from .agent_micro import AgentMicroLMModel
from .loaders import _lm_kwargs


_AGENT_LANE_PREFIXES = (
    "action_backbone_adapter.",
    "action_emb.",
    "environment_emb.",
    "action_linear.",
    "action_to_depformer.",
    "action_history_emb.",
    "environment_history_emb.",
    "response_environment_history_emb.",
    "response_environment_scale",
    "response_action_scale",
    "response_temporal_base_in.",
    "response_temporal_context_in.",
    "response_temporal_gru.",
    "response_temporal_out.",
    "response_eos_head.",
    "response_token_head.",
    "response_token_context_in.",
    "response_token_repair_down.",
    "response_token_repair_up.",
    "response_token_repair_gate.",
    "response_token_continuation_down.",
    "response_token_continuation_up.",
    "action_depth_in.",
    "action_temporal_in.",
    "action_temporal_gru.",
    "action_call_gate_in.",
    "action_call_gate_text_in.",
    "action_call_gate_gru.",
    "action_call_gate_phase_emb.",
    "action_phase_emb.",
    "action_call_gate.",
    "action_call_intent_in.",
    "action_call_intent_text_in.",
    "action_call_intent_gru.",
    "action_call_intent_norm.",
    "action_call_intent.",
    "action_call_soft_intent_norm.",
    "action_call_soft_intent.",
    "action_depth_text_emb.",
    "action_depth_emb.",
    "action_depth.",
    "action_depth_linears.",
    "action_timer_argument_linear.",
    "action_timer_argument_mlp_in.",
    "action_timer_argument_mlp_norm.",
    "action_timer_argument_mlp_out.",
    "action_start_depth_in.",
    "action_start_temporal_in.",
    "action_start_temporal_gru.",
    "action_start_phase_emb.",
    "action_start_depth_text_emb.",
    "action_start_depth.",
    "action_start_linear.",
    "action_summary_proj.",
    "action_audio_conditioning_gate",
)

_OPTIONAL_MIGRATION_LANE_PREFIX_GROUPS = (
    ("action_backbone_adapter.",),
    ("action_temporal_in.", "action_temporal_gru."),
    ("action_call_gate.",),
    ("action_call_intent.",),
    (
        "action_call_intent_in.",
        "action_call_intent_text_in.",
        "action_call_intent_gru.",
        "action_call_intent_norm.",
    ),
    ("action_call_soft_intent_norm.", "action_call_soft_intent."),
    (
        "action_call_gate_in.",
        "action_call_gate_text_in.",
        "action_call_gate_gru.",
        "action_call_gate_phase_emb.",
    ),
    (
        "action_start_depth_in.",
        "action_start_temporal_in.",
        "action_start_temporal_gru.",
        "action_start_phase_emb.",
        "action_start_depth_text_emb.",
        "action_start_depth.",
        "action_start_linear.",
    ),
    ("action_timer_argument_linear.",),
    (
        "action_timer_argument_mlp_in.",
        "action_timer_argument_mlp_norm.",
        "action_timer_argument_mlp_out.",
    ),
    ("response_environment_history_emb.", "response_environment_scale"),
    ("response_action_scale",),
    (
        "response_temporal_base_in.",
        "response_temporal_context_in.",
        "response_temporal_gru.",
        "response_temporal_out.",
    ),
    ("response_eos_head.",),
    ("response_token_head.",),
    ("response_token_context_in.",),
    ("response_token_repair_down.", "response_token_repair_up."),
    ("response_token_repair_gate.",),
    (
        "response_token_continuation_down.",
        "response_token_continuation_up.",
    ),
    ("action_audio_conditioning_gate",),
)

_OPTIONAL_MIGRATION_LANE_PREFIXES = tuple(
    prefix
    for group in _OPTIONAL_MIGRATION_LANE_PREFIX_GROUPS
    for prefix in group
)


def _patch_personaplex_state_dict(
    state_dict: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
    *,
    copy_missing_weights: bool,
) -> None:
    """Apply the same 8-to-16-codebook compatibility patches as upstream."""

    for name, tensor in list(state_dict.items()):
        if "depformer" in name and "self_attn" in name and name in model_state:
            if tensor.shape != model_state[name].shape:
                missing = tensor if copy_missing_weights else model_state[name][tensor.shape[0] :]
                state_dict[name] = torch.concat([tensor, missing], dim=0)

    if not copy_missing_weights:
        return
    to_replace = ("gating", "linears", "depformer_in", "depformer_emb")
    for name in model_state:
        if name in state_dict or name.startswith(_AGENT_LANE_PREFIXES):
            continue
        replaced = False
        for old, new in zip(range(8), range(8, 16)):
            for group in to_replace:
                needle = f"{group}.{new}."
                if needle not in name:
                    continue
                source = name.replace(needle, f"{group}.{old}.")
                if source in state_dict:
                    state_dict[name] = state_dict[source]
                    replaced = True
                break
            if replaced:
                break


def _initialize_missing_agent_parameters(
    model: AgentLMModel,
    device: torch.device,
    missing_keys: list[str],
) -> None:
    """Initialize only new parameters absent from the supplied checkpoint.

    A full agent checkpoint may already contain trained lane weights.  Those
    parameters are assigned by ``load_state_dict`` and must never be reset.
    """

    missing = set(missing_keys)
    for module_name, module in model.named_modules():
        direct_parameters = list(module.named_parameters(recurse=False))
        if not direct_parameters:
            continue
        parameter_keys = [
            f"{module_name}.{name}" if module_name else name
            for name, _ in direct_parameters
        ]
        missing_parameter_keys = [key for key in parameter_keys if key in missing]
        if not missing_parameter_keys:
            continue
        if len(missing_parameter_keys) != len(parameter_keys):
            raise RuntimeError(
                "checkpoint contains a partial agent module; refusing to reset "
                f"loaded parameters in {module_name!r}"
            )
        module.to_empty(device=device, recurse=False)
        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            reset_parameters()
            continue
        # A few Moshi streaming primitives own parameters directly but do not
        # expose reset_parameters (notably RMSNorm and streaming MHA).  Apply
        # the constructor-equivalent safe defaults to those missing tensors.
        with torch.no_grad():
            for parameter_name, parameter in module.named_parameters(recurse=False):
                if parameter_name in {"alpha", "weight"} and parameter.dim() <= 1:
                    parameter.fill_(1.0)
                elif parameter.dim() >= 2:
                    torch.nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
                else:
                    parameter.zero_()


def get_personaplex_agent_lm(
    filename: str | Path | None,
    *,
    action_card: int = 256,
    environment_card: int = 256,
    copy_missing_weights: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    delays: list[int] | None = None,
    micro_decoder: bool = False,
    action_end_token_id: int = 4,
    action_call_begin_token_id: int = 1,
    action_call_gate_threshold: float = 0.0,
    action_timer_tool_token_id: int = 14,
    action_depth_dim: int = 512,
    action_depth_num_heads: int = 8,
    action_depth_num_layers: int = 2,
    action_backbone_adapter_dim: int = 64,
) -> AgentLMModel | AgentMicroLMModel:
    """Load base PersonaPlex weights plus fresh native typed-lane parameters.

    ``filename`` may be ``None`` for a randomly initialized development model.
    When a base checkpoint is supplied, all its weights are loaded exactly as
    upstream does; only action/result lane modules are newly initialized.
    """

    lm_kwargs = dict(_lm_kwargs)
    lm_kwargs["dep_q"] = 16
    if delays is not None:
        lm_kwargs["delays"] = delays

    target_device = torch.device(device)
    init_device: torch.device | str = "meta" if filename is not None else target_device
    model_class = AgentMicroLMModel if micro_decoder else AgentLMModel
    micro_kwargs = (
        {
            "action_end_token_id": action_end_token_id,
            "action_call_begin_token_id": action_call_begin_token_id,
            "action_call_gate_threshold": action_call_gate_threshold,
            "action_timer_tool_token_id": action_timer_tool_token_id,
            "action_depth_dim": action_depth_dim,
            "action_depth_num_heads": action_depth_num_heads,
            "action_depth_num_layers": action_depth_num_layers,
            "action_backbone_adapter_dim": action_backbone_adapter_dim,
        }
        if micro_decoder
        else {}
    )
    model = model_class(
        device=init_device,
        dtype=dtype,
        action_card=action_card,
        environment_card=environment_card,
        **micro_kwargs,
        **lm_kwargs,
    )
    if filename is None:
        return model.to(device=target_device, dtype=dtype).eval()

    checkpoint_path = Path(filename)
    if checkpoint_path.suffix in (".safetensors", ".sft", ".sfts"):
        # Load on CPU first.  The upstream loader uses ``dev.type`` which can
        # resolve ``cuda:3`` to GPU 0; this preserves the caller's explicit GPU
        # selection and avoids disturbing another local model.
        state_dict = load_file(checkpoint_path, device="cpu")
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict) and isinstance(state_dict.get("model"), dict):
            state_dict = state_dict["model"]

    _patch_personaplex_state_dict(
        state_dict, model.state_dict(), copy_missing_weights=copy_missing_weights
    )
    for key, tensor in state_dict.items():
        state_dict[key] = tensor.to(device=target_device, dtype=dtype)

    incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
    missing_base = [
        key for key in incompatible.missing_keys if not key.startswith(_AGENT_LANE_PREFIXES)
    ]
    if missing_base:
        raise RuntimeError(f"base checkpoint is missing required weights: {missing_base}")
    if incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint has unexpected weights: {incompatible.unexpected_keys}")

    _initialize_missing_agent_parameters(
        model, target_device, incompatible.missing_keys
    )
    return model.to(device=target_device, dtype=dtype).eval()


def get_personaplex_agent_micro_lm(
    filename: str | Path | None,
    **kwargs,
) -> AgentMicroLMModel:
    """Convenience wrapper for the five-slot v2 action micro-decoder."""

    model = get_personaplex_agent_lm(filename, micro_decoder=True, **kwargs)
    if not isinstance(model, AgentMicroLMModel):
        raise TypeError("micro-decoder loader returned the wrong model type")
    return model
