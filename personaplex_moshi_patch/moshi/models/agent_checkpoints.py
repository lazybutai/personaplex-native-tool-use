"""Small standalone checkpoints for PersonaPlex-Agent lane parameters."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from .agent_lm import AgentLMModel
from .agent_loaders import (
    _AGENT_LANE_PREFIXES,
    _OPTIONAL_MIGRATION_LANE_PREFIXES,
    _OPTIONAL_MIGRATION_LANE_PREFIX_GROUPS,
)


AGENT_LANE_CHECKPOINT_FORMAT = "personaplex-agent-lanes-v1"


def agent_lane_state_dict(
    model: AgentLMModel,
    *,
    include_disabled_optional: bool = False,
) -> dict[str, torch.Tensor]:
    """Return only newly introduced lane tensors, never base PersonaPlex weights."""

    state = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if name.startswith(_AGENT_LANE_PREFIXES)
    }
    # The EOS head is an optional migration lane. A model that loaded a
    # pre-EOS checkpoint still owns an initialized module in memory, but an
    # unrelated trainer must not silently add that disabled head when saving.
    if (
        not include_disabled_optional
        and not getattr(model, "response_eos_conditioning_enabled", False)
    ):
        state = {
            name: tensor
            for name, tensor in state.items()
            if not name.startswith("response_eos_head.")
        }
    if (
        not include_disabled_optional
        and not getattr(model, "response_action_conditioning_enabled", False)
    ):
        state = {
            name: tensor
            for name, tensor in state.items()
            if not name.startswith("response_action_scale")
        }
    if (
        not include_disabled_optional
        and not getattr(model, "response_token_conditioning_enabled", False)
    ):
        state = {
            name: tensor
            for name, tensor in state.items()
            if not name.startswith("response_token_head.")
        }
    if (
        not include_disabled_optional
        and not getattr(
            model, "response_token_context_conditioning_enabled", False
        )
    ):
        state = {
            name: tensor
            for name, tensor in state.items()
            if not name.startswith("response_token_context_in.")
        }
    if (
        not include_disabled_optional
        and not getattr(
            model, "response_token_repair_conditioning_enabled", False
        )
    ):
        state = {
            name: tensor
            for name, tensor in state.items()
            if not name.startswith(
                ("response_token_repair_down.", "response_token_repair_up.")
            )
        }
    if (
        not include_disabled_optional
        and not getattr(
            model,
            "response_token_continuation_conditioning_enabled",
            False,
        )
    ):
        state = {
            name: tensor
            for name, tensor in state.items()
            if not name.startswith(
                (
                    "response_token_continuation_down.",
                    "response_token_continuation_up.",
                )
            )
        }
    if (
        not include_disabled_optional
        and not getattr(
            model, "response_token_repair_gate_conditioning_enabled", False
        )
    ):
        state = {
            name: tensor
            for name, tensor in state.items()
            if not name.startswith("response_token_repair_gate.")
        }
    if (
        not include_disabled_optional
        and not getattr(model, "action_timer_argument_conditioning_enabled", False)
    ):
        state = {
            name: tensor
            for name, tensor in state.items()
            if not name.startswith("action_timer_argument_linear.")
        }
    if (
        not include_disabled_optional
        and not getattr(
            model, "action_timer_argument_mlp_conditioning_enabled", False
        )
    ):
        state = {
            name: tensor
            for name, tensor in state.items()
            if not name.startswith(
                (
                    "action_timer_argument_mlp_in.",
                    "action_timer_argument_mlp_norm.",
                    "action_timer_argument_mlp_out.",
                )
            )
        }
    if not state:
        raise ValueError("model has no PersonaPlex-Agent lane tensors")
    return state


def save_agent_lane_checkpoint(
    model: AgentLMModel,
    filename: str | Path,
    *,
    action_manifest_fingerprint: str,
    base_checkpoint_id: str = "nvidia/personaplex-7b-v1",
    extra_metadata: Mapping[str, str] | None = None,
) -> Path:
    """Save a compact adapter-style checkpoint with protocol identity metadata."""

    if not action_manifest_fingerprint:
        raise ValueError("action manifest fingerprint is required")
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": AGENT_LANE_CHECKPOINT_FORMAT,
        "architecture": type(model).__name__,
        "action_manifest_fingerprint": action_manifest_fingerprint,
        "base_checkpoint_id": base_checkpoint_id,
        "action_card": str(model.action_card),
        "environment_card": str(model.environment_card),
    }
    capability_metadata = {
        "separate_action_start_branch": str(
            getattr(model, "separate_action_start_branch_enabled", False)
        ).lower(),
        "action_timer_argument_conditioning": str(
            getattr(model, "action_timer_argument_conditioning_enabled", False)
        ).lower(),
        "action_timer_argument_mlp_conditioning": str(
            getattr(model, "action_timer_argument_mlp_conditioning_enabled", False)
        ).lower(),
        "response_environment_conditioning": str(
            getattr(model, "response_environment_conditioning_enabled", False)
        ).lower(),
        "response_action_conditioning": str(
            getattr(model, "response_action_conditioning_enabled", False)
        ).lower(),
        "response_temporal_conditioning": str(
            getattr(model, "response_temporal_conditioning_enabled", False)
        ).lower(),
        "response_eos_conditioning": str(
            getattr(model, "response_eos_conditioning_enabled", False)
        ).lower(),
        "response_token_conditioning": str(
            getattr(model, "response_token_conditioning_enabled", False)
        ).lower(),
        "response_token_context_conditioning": str(
            getattr(
                model,
                "response_token_context_conditioning_enabled",
                False,
            )
        ).lower(),
        "response_token_repair_conditioning": str(
            getattr(
                model,
                "response_token_repair_conditioning_enabled",
                False,
            )
        ).lower(),
        "response_token_continuation_conditioning": str(
            getattr(
                model,
                "response_token_continuation_conditioning_enabled",
                False,
            )
        ).lower(),
        "response_token_repair_gate_conditioning": str(
            getattr(
                model,
                "response_token_repair_gate_conditioning_enabled",
                False,
            )
        ).lower(),
        "response_token_repair_nonnegative_rows": ",".join(
            str(row_id)
            for row_id in getattr(
                model, "response_token_repair_nonnegative_rows", ()
            )
        ),
        "response_token_repair_scope_environment_token_ids": ",".join(
            str(token_id)
            for token_id in getattr(
                model,
                "response_token_repair_scope_environment_token_ids",
                (),
            )
        ),
        "response_token_repair_scope_until_first_semantic_token": str(
            getattr(
                model,
                "response_token_repair_scope_until_first_semantic_token",
                False,
            )
        ).lower(),
        "response_token_repair_onset_row_biases": ",".join(
            f"{row_id}:{format(bias, '.17g')}"
            for row_id, bias in getattr(
                model,
                "response_token_repair_onset_row_biases",
                (),
            )
        ),
        "response_token_repair_post_onset_disabled_rows": ",".join(
            str(row_id)
            for row_id in getattr(
                model,
                "response_token_repair_post_onset_disabled_rows",
                (),
            )
        ),
        "response_token_repair_post_onset_row_biases": ",".join(
            f"{row_id}:{format(bias, '.17g')}"
            for row_id, bias in getattr(
                model,
                "response_token_repair_post_onset_row_biases",
                (),
            )
        ),
        "response_token_repair_onset_generated_semantic_latch": str(
            getattr(
                model,
                "response_token_repair_onset_generated_semantic_latch",
                False,
            )
        ).lower(),
    }
    if extra_metadata:
        for key, value in extra_metadata.items():
            if key in metadata:
                raise ValueError(f"extra metadata cannot override reserved key {key!r}")
            metadata[str(key)] = str(value)
    for key, value in capability_metadata.items():
        metadata.setdefault(key, value)
    cpu_state = {
        name: tensor.detach().to(device="cpu").contiguous()
        for name, tensor in agent_lane_state_dict(model).items()
    }
    save_file(cpu_state, str(path), metadata=metadata)
    return path


def load_agent_lane_checkpoint(
    model: AgentLMModel,
    filename: str | Path,
    *,
    expected_action_manifest_fingerprint: str,
    strict: bool = True,
) -> dict[str, str]:
    """Load trained lane tensors without replacing any base PersonaPlex weight."""

    path = Path(filename)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    if metadata.get("format") != AGENT_LANE_CHECKPOINT_FORMAT:
        raise ValueError("not a PersonaPlex-Agent lane checkpoint")
    if metadata.get("action_manifest_fingerprint") != expected_action_manifest_fingerprint:
        raise ValueError("action manifest fingerprint mismatch")
    if metadata.get("action_card") != str(model.action_card) or metadata.get(
        "environment_card"
    ) != str(model.environment_card):
        raise ValueError("lane checkpoint vocabulary cardinality mismatch")

    state = load_file(str(path), device="cpu")
    if any(not name.startswith(_AGENT_LANE_PREFIXES) for name in state):
        raise ValueError("lane checkpoint contains a non-agent/base-model tensor")
    expected = set(agent_lane_state_dict(model, include_disabled_optional=True))
    provided = set(state)
    optional_expected = {
        name
        for name in expected
        if name.startswith(_OPTIONAL_MIGRATION_LANE_PREFIXES)
    }
    if strict and expected != provided:
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        missing_required = sorted(set(missing) - optional_expected)
        partial_optional_groups = []
        for prefixes in _OPTIONAL_MIGRATION_LANE_PREFIX_GROUPS:
            group_expected = {
                name for name in optional_expected if name.startswith(prefixes)
            }
            group_provided = provided & group_expected
            if group_provided and group_provided != group_expected:
                partial_optional_groups.append(prefixes)
        if missing_required or unexpected or partial_optional_groups:
            raise ValueError(
                "lane checkpoint key mismatch; "
                f"missing={missing[:3]}, unexpected={unexpected[:3]}, "
                f"partial_optional_groups={partial_optional_groups}"
            )
    missing_optional = optional_expected - provided
    if missing_optional:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in missing_optional:
                    parameter.zero_()
        reportable_missing = set(missing_optional)
        if metadata.get("response_eos_conditioning", "false").lower() == "false":
            reportable_missing = {
                name
                for name in reportable_missing
                if not name.startswith("response_eos_head.")
            }
        if (
            metadata.get("response_action_conditioning", "false").lower()
            == "false"
        ):
            reportable_missing = {
                name
                for name in reportable_missing
                if not name.startswith("response_action_scale")
            }
        if (
            metadata.get("response_token_conditioning", "false").lower()
            == "false"
        ):
            reportable_missing = {
                name
                for name in reportable_missing
                if not name.startswith("response_token_head.")
            }
        if (
            metadata.get(
                "response_token_context_conditioning", "false"
            ).lower()
            == "false"
        ):
            reportable_missing = {
                name
                for name in reportable_missing
                if not name.startswith("response_token_context_in.")
            }
        if (
            metadata.get(
                "response_token_repair_conditioning", "false"
            ).lower()
            == "false"
        ):
            reportable_missing = {
                name
                for name in reportable_missing
                if not name.startswith(
                    ("response_token_repair_down.", "response_token_repair_up.")
                )
            }
        if (
            metadata.get(
                "response_token_continuation_conditioning", "false"
            ).lower()
            == "false"
        ):
            reportable_missing = {
                name
                for name in reportable_missing
                if not name.startswith(
                    (
                        "response_token_continuation_down.",
                        "response_token_continuation_up.",
                    )
                )
            }
        if (
            metadata.get(
                "response_token_repair_gate_conditioning", "false"
            ).lower()
            == "false"
        ):
            reportable_missing = {
                name
                for name in reportable_missing
                if not name.startswith("response_token_repair_gate.")
            }
        if (
            metadata.get("action_timer_argument_conditioning", "false").lower()
            == "false"
        ):
            reportable_missing = {
                name
                for name in reportable_missing
                if not name.startswith("action_timer_argument_linear.")
            }
        if (
            metadata.get(
                "action_timer_argument_mlp_conditioning", "false"
            ).lower()
            == "false"
        ):
            reportable_missing = {
                name
                for name in reportable_missing
                if not name.startswith(
                    (
                        "action_timer_argument_mlp_in.",
                        "action_timer_argument_mlp_norm.",
                        "action_timer_argument_mlp_out.",
                    )
                )
            }
        missing_groups = []
        for prefixes in _OPTIONAL_MIGRATION_LANE_PREFIX_GROUPS:
            if any(name.startswith(prefixes) for name in reportable_missing):
                missing_groups.append("+".join(prefix.rstrip(".") for prefix in prefixes))
        if missing_groups:
            metadata["optional_lane_migration"] = (
                "missing-optional-lanes-zero-disabled:" + ",".join(missing_groups)
            )
    model_state = model.state_dict()
    for name, tensor in list(state.items()):
        if name not in model_state:
            if strict:
                raise ValueError(f"unknown lane tensor {name!r}")
            del state[name]
            continue
        if tensor.shape != model_state[name].shape:
            raise ValueError(f"lane tensor shape mismatch for {name!r}")
        state[name] = tensor.to(
            device=model_state[name].device, dtype=model_state[name].dtype
        )
    incompatible = model.load_state_dict(state, strict=False, assign=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected lane tensors: {incompatible.unexpected_keys}")
    missing_agent = [
        name
        for name in incompatible.missing_keys
        if name.startswith(_AGENT_LANE_PREFIXES) and name in provided
    ]
    if missing_agent:
        raise RuntimeError(f"failed to load lane tensors: {missing_agent}")
    calibrated_threshold = metadata.get("calibrated_call_gate_threshold")
    if calibrated_threshold is not None:
        try:
            threshold = float(calibrated_threshold)
        except ValueError as error:
            raise ValueError("lane checkpoint has an invalid calibrated threshold") from error
        if not math.isfinite(threshold):
            raise ValueError("lane checkpoint calibrated threshold must be finite")
        if not hasattr(model, "action_call_gate_threshold"):
            raise ValueError("calibrated threshold requires a micro action model")
        model.action_call_gate_threshold = threshold
    start_branch_value = metadata.get("separate_action_start_branch", "false").lower()
    if start_branch_value not in {"true", "false"}:
        raise ValueError("lane checkpoint has an invalid separate start-branch flag")
    if start_branch_value == "true":
        if not hasattr(model, "separate_action_start_branch_enabled"):
            raise ValueError("separate start branch requires a micro action model")
        required_start = {
            name for name in model_state if name.startswith("action_start_")
        }
        provided_start = provided & required_start
        if not required_start or provided_start != required_start:
            missing_start = sorted(required_start - provided_start)
            raise ValueError(
                "enabled separate start branch is incomplete; "
                f"missing={missing_start[:3]}"
            )
        model.separate_action_start_branch_enabled = True
    elif hasattr(model, "separate_action_start_branch_enabled"):
        model.separate_action_start_branch_enabled = False
    timer_argument_value = metadata.get(
        "action_timer_argument_conditioning", "false"
    ).lower()
    if timer_argument_value not in {"true", "false"}:
        raise ValueError("lane checkpoint has an invalid TIMER-argument flag")
    if timer_argument_value == "true":
        if not hasattr(model, "action_timer_argument_conditioning_enabled"):
            raise ValueError("TIMER argument conditioning requires a micro action model")
        required_timer_argument = {
            name
            for name in model_state
            if name.startswith("action_timer_argument_linear.")
        }
        provided_timer_argument = provided & required_timer_argument
        if (
            not required_timer_argument
            or provided_timer_argument != required_timer_argument
        ):
            missing_timer_argument = sorted(
                required_timer_argument - provided_timer_argument
            )
            raise ValueError(
                "enabled TIMER argument conditioning is incomplete; "
                f"missing={missing_timer_argument[:3]}"
            )
        model.action_timer_argument_conditioning_enabled = True
    elif hasattr(model, "action_timer_argument_conditioning_enabled"):
        model.action_timer_argument_conditioning_enabled = False
    timer_argument_mlp_value = metadata.get(
        "action_timer_argument_mlp_conditioning", "false"
    ).lower()
    if timer_argument_mlp_value not in {"true", "false"}:
        raise ValueError("lane checkpoint has an invalid TIMER-argument MLP flag")
    if timer_argument_mlp_value == "true":
        if not hasattr(model, "action_timer_argument_mlp_conditioning_enabled"):
            raise ValueError("TIMER argument MLP requires a micro action model")
        timer_argument_mlp_prefixes = (
            "action_timer_argument_mlp_in.",
            "action_timer_argument_mlp_norm.",
            "action_timer_argument_mlp_out.",
        )
        required_timer_argument_mlp = {
            name for name in model_state if name.startswith(timer_argument_mlp_prefixes)
        }
        provided_timer_argument_mlp = provided & required_timer_argument_mlp
        if (
            not required_timer_argument_mlp
            or provided_timer_argument_mlp != required_timer_argument_mlp
        ):
            missing_timer_argument_mlp = sorted(
                required_timer_argument_mlp - provided_timer_argument_mlp
            )
            raise ValueError(
                "enabled TIMER argument MLP is incomplete; "
                f"missing={missing_timer_argument_mlp[:3]}"
            )
        model.action_timer_argument_mlp_conditioning_enabled = True
    elif hasattr(model, "action_timer_argument_mlp_conditioning_enabled"):
        model.action_timer_argument_mlp_conditioning_enabled = False
    response_value = metadata.get(
        "response_environment_conditioning", "false"
    ).lower()
    if response_value not in {"true", "false"}:
        raise ValueError(
            "lane checkpoint has an invalid response-environment flag"
        )
    if response_value == "true":
        if not hasattr(model, "response_environment_conditioning_enabled"):
            raise ValueError(
                "response-environment conditioning requires a micro action model"
            )
        required_response = {
            name
            for name in model_state
            if name.startswith(
                ("response_environment_history_emb.", "response_environment_scale")
            )
        }
        provided_response = provided & required_response
        if not required_response or provided_response != required_response:
            missing_response = sorted(required_response - provided_response)
            raise ValueError(
                "enabled response-environment conditioning is incomplete; "
                f"missing={missing_response[:3]}"
            )
        model.response_environment_conditioning_enabled = True
    elif hasattr(model, "response_environment_conditioning_enabled"):
        model.response_environment_conditioning_enabled = False
    temporal_response_value = metadata.get(
        "response_temporal_conditioning", "false"
    ).lower()
    if temporal_response_value not in {"true", "false"}:
        raise ValueError("lane checkpoint has an invalid temporal-response flag")
    if temporal_response_value == "true":
        if not hasattr(model, "response_temporal_conditioning_enabled"):
            raise ValueError("temporal response requires a micro action model")
        temporal_prefixes = (
            "response_temporal_base_in.",
            "response_temporal_context_in.",
            "response_temporal_gru.",
            "response_temporal_out.",
        )
        required_temporal = {
            name for name in model_state if name.startswith(temporal_prefixes)
        }
        provided_temporal = provided & required_temporal
        if not required_temporal or provided_temporal != required_temporal:
            missing_temporal = sorted(required_temporal - provided_temporal)
            raise ValueError(
                "enabled temporal response is incomplete; "
                f"missing={missing_temporal[:3]}"
            )
        if not model.response_environment_conditioning_enabled:
            raise ValueError(
                "temporal response requires response-environment conditioning"
            )
        model.response_temporal_conditioning_enabled = True
    elif hasattr(model, "response_temporal_conditioning_enabled"):
        model.response_temporal_conditioning_enabled = False
    response_action_value = metadata.get(
        "response_action_conditioning", "false"
    ).lower()
    if response_action_value not in {"true", "false"}:
        raise ValueError("lane checkpoint has an invalid response-action flag")
    if response_action_value == "true":
        if not hasattr(model, "response_action_conditioning_enabled"):
            raise ValueError("response action conditioning requires a micro action model")
        required_response_action = {
            name for name in model_state if name.startswith("response_action_scale")
        }
        provided_response_action = provided & required_response_action
        if (
            not required_response_action
            or provided_response_action != required_response_action
        ):
            missing_response_action = sorted(
                required_response_action - provided_response_action
            )
            raise ValueError(
                "enabled response action conditioning is incomplete; "
                f"missing={missing_response_action[:3]}"
            )
        if not model.response_environment_conditioning_enabled:
            raise ValueError(
                "response action conditioning requires response-environment conditioning"
            )
        if not model.response_temporal_conditioning_enabled:
            raise ValueError(
                "response action conditioning requires temporal response conditioning"
            )
        model.response_action_conditioning_enabled = True
    elif hasattr(model, "response_action_conditioning_enabled"):
        model.response_action_conditioning_enabled = False
    response_token_value = metadata.get(
        "response_token_conditioning", "false"
    ).lower()
    if response_token_value not in {"true", "false"}:
        raise ValueError("lane checkpoint has an invalid response-token flag")
    if response_token_value == "true":
        if not hasattr(model, "response_token_conditioning_enabled"):
            raise ValueError("response token conditioning requires a micro action model")
        required_response_token = {
            name for name in model_state if name.startswith("response_token_head.")
        }
        provided_response_token = provided & required_response_token
        if (
            not required_response_token
            or provided_response_token != required_response_token
        ):
            missing_response_token = sorted(
                required_response_token - provided_response_token
            )
            raise ValueError(
                "enabled response token conditioning is incomplete; "
                f"missing={missing_response_token[:3]}"
            )
        if not model.response_temporal_conditioning_enabled:
            raise ValueError(
                "response token conditioning requires temporal response conditioning"
            )
        model.response_token_conditioning_enabled = True
    elif hasattr(model, "response_token_conditioning_enabled"):
        model.response_token_conditioning_enabled = False
    response_token_context_value = metadata.get(
        "response_token_context_conditioning", "false"
    ).lower()
    if response_token_context_value not in {"true", "false"}:
        raise ValueError(
            "lane checkpoint has an invalid response-token-context flag"
        )
    if response_token_context_value == "true":
        if not hasattr(model, "response_token_context_conditioning_enabled"):
            raise ValueError(
                "response token context requires a micro action model"
            )
        required_response_token_context = {
            name
            for name in model_state
            if name.startswith("response_token_context_in.")
        }
        provided_response_token_context = (
            provided & required_response_token_context
        )
        if (
            not required_response_token_context
            or provided_response_token_context
            != required_response_token_context
        ):
            missing_response_token_context = sorted(
                required_response_token_context
                - provided_response_token_context
            )
            raise ValueError(
                "enabled response token context is incomplete; "
                f"missing={missing_response_token_context[:3]}"
            )
        if not model.response_temporal_conditioning_enabled:
            raise ValueError(
                "response token context requires temporal response conditioning"
            )
        if not model.response_token_conditioning_enabled:
            raise ValueError(
                "response token context requires response token conditioning"
            )
        model.response_token_context_conditioning_enabled = True
    elif hasattr(model, "response_token_context_conditioning_enabled"):
        model.response_token_context_conditioning_enabled = False
    response_token_repair_value = metadata.get(
        "response_token_repair_conditioning", "false"
    ).lower()
    if response_token_repair_value not in {"true", "false"}:
        raise ValueError(
            "lane checkpoint has an invalid response-token-repair flag"
        )
    if response_token_repair_value == "true":
        if not hasattr(model, "response_token_repair_conditioning_enabled"):
            raise ValueError(
                "response token repair requires a micro action model"
            )
        repair_prefixes = (
            "response_token_repair_down.",
            "response_token_repair_up.",
        )
        required_response_token_repair = {
            name for name in model_state if name.startswith(repair_prefixes)
        }
        provided_response_token_repair = (
            provided & required_response_token_repair
        )
        if (
            not required_response_token_repair
            or provided_response_token_repair
            != required_response_token_repair
        ):
            missing_response_token_repair = sorted(
                required_response_token_repair
                - provided_response_token_repair
            )
            raise ValueError(
                "enabled response token repair is incomplete; "
                f"missing={missing_response_token_repair[:3]}"
            )
        if not model.response_temporal_conditioning_enabled:
            raise ValueError(
                "response token repair requires temporal response conditioning"
            )
        if not model.response_token_conditioning_enabled:
            raise ValueError(
                "response token repair requires response token conditioning"
            )
        model.response_token_repair_conditioning_enabled = True
    elif hasattr(model, "response_token_repair_conditioning_enabled"):
        model.response_token_repair_conditioning_enabled = False
    response_token_repair_gate_value = metadata.get(
        "response_token_repair_gate_conditioning", "false"
    ).lower()
    if response_token_repair_gate_value not in {"true", "false"}:
        raise ValueError(
            "lane checkpoint has an invalid response-token-repair-gate flag"
        )
    if response_token_repair_gate_value == "true":
        if not hasattr(
            model, "response_token_repair_gate_conditioning_enabled"
        ):
            raise ValueError("response repair gate requires a micro action model")
        required_response_token_repair_gate = {
            name
            for name in model_state
            if name.startswith("response_token_repair_gate.")
        }
        provided_response_token_repair_gate = (
            provided & required_response_token_repair_gate
        )
        if (
            not required_response_token_repair_gate
            or provided_response_token_repair_gate
            != required_response_token_repair_gate
        ):
            missing_response_token_repair_gate = sorted(
                required_response_token_repair_gate
                - provided_response_token_repair_gate
            )
            raise ValueError(
                "enabled response token repair gate is incomplete; "
                f"missing={missing_response_token_repair_gate[:3]}"
            )
        if not model.response_token_repair_conditioning_enabled:
            raise ValueError("response repair gate requires response token repair")
        model.response_token_repair_gate_conditioning_enabled = True
    elif hasattr(model, "response_token_repair_gate_conditioning_enabled"):
        model.response_token_repair_gate_conditioning_enabled = False
    nonnegative_rows_text = metadata.get(
        "response_token_repair_nonnegative_rows", ""
    )
    try:
        nonnegative_rows = tuple(
            int(value) for value in nonnegative_rows_text.split(",") if value
        )
    except ValueError as error:
        raise ValueError(
            "lane checkpoint has invalid nonnegative repair rows"
        ) from error
    if ",".join(str(value) for value in nonnegative_rows) != nonnegative_rows_text:
        raise ValueError(
            "lane checkpoint nonnegative repair rows must be canonical"
        )
    if tuple(sorted(set(nonnegative_rows))) != nonnegative_rows:
        raise ValueError(
            "lane checkpoint nonnegative repair rows must be sorted and unique"
        )
    if not hasattr(model, "set_response_token_repair_nonnegative_rows"):
        if nonnegative_rows:
            raise ValueError(
                "nonnegative repair rows require a micro action model"
            )
    else:
        model.set_response_token_repair_nonnegative_rows(nonnegative_rows)
    repair_scope_text = metadata.get(
        "response_token_repair_scope_environment_token_ids", ""
    )
    try:
        repair_scope_token_ids = tuple(
            int(value) for value in repair_scope_text.split(",") if value
        )
    except ValueError as error:
        raise ValueError(
            "lane checkpoint has invalid response repair scope tokens"
        ) from error
    if (
        ",".join(str(value) for value in repair_scope_token_ids)
        != repair_scope_text
    ):
        raise ValueError(
            "lane checkpoint response repair scope tokens must be canonical"
        )
    if tuple(sorted(set(repair_scope_token_ids))) != repair_scope_token_ids:
        raise ValueError(
            "lane checkpoint response repair scope tokens must be sorted and unique"
        )
    if not hasattr(
        model, "set_response_token_repair_scope_environment_token_ids"
    ):
        if repair_scope_token_ids:
            raise ValueError(
                "response repair scope tokens require a micro action model"
            )
    else:
        model.set_response_token_repair_scope_environment_token_ids(
            repair_scope_token_ids
        )
    response_token_continuation_value = metadata.get(
        "response_token_continuation_conditioning", "false"
    ).lower()
    if response_token_continuation_value not in {"true", "false"}:
        raise ValueError(
            "lane checkpoint has an invalid response-token-continuation flag"
        )
    if response_token_continuation_value == "true":
        if not hasattr(
            model, "response_token_continuation_conditioning_enabled"
        ):
            raise ValueError(
                "response token continuation requires a micro action model"
            )
        continuation_prefixes = (
            "response_token_continuation_down.",
            "response_token_continuation_up.",
        )
        required_response_token_continuation = {
            name
            for name in model_state
            if name.startswith(continuation_prefixes)
        }
        provided_response_token_continuation = (
            provided & required_response_token_continuation
        )
        if (
            not required_response_token_continuation
            or provided_response_token_continuation
            != required_response_token_continuation
        ):
            missing_response_token_continuation = sorted(
                required_response_token_continuation
                - provided_response_token_continuation
            )
            raise ValueError(
                "enabled response token continuation is incomplete; "
                f"missing={missing_response_token_continuation[:3]}"
            )
        if not model.response_temporal_conditioning_enabled:
            raise ValueError(
                "response token continuation requires temporal response "
                "conditioning"
            )
        if not model.response_token_conditioning_enabled:
            raise ValueError(
                "response token continuation requires response token "
                "conditioning"
            )
        if not model.response_token_repair_conditioning_enabled:
            raise ValueError(
                "response token continuation requires response token repair"
            )
        if not repair_scope_token_ids:
            raise ValueError(
                "response token continuation requires an executor result scope"
            )
        model.response_token_continuation_conditioning_enabled = True
    elif hasattr(
        model, "response_token_continuation_conditioning_enabled"
    ):
        model.response_token_continuation_conditioning_enabled = False
    onset_bias_text = metadata.get(
        "response_token_repair_onset_row_biases", ""
    )
    onset_bias_pairs: list[tuple[int, float]] = []
    try:
        for item in onset_bias_text.split(","):
            if not item:
                continue
            row_text, bias_text = item.split(":", maxsplit=1)
            onset_bias_pairs.append((int(row_text), float(bias_text)))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "lane checkpoint has invalid response repair onset row biases"
        ) from error
    canonical_onset_bias_text = ",".join(
        f"{row_id}:{format(bias, '.17g')}"
        for row_id, bias in onset_bias_pairs
    )
    if canonical_onset_bias_text != onset_bias_text:
        raise ValueError(
            "lane checkpoint response repair onset row biases must be canonical"
        )
    if tuple(sorted(onset_bias_pairs)) != tuple(onset_bias_pairs) or len(
        {row_id for row_id, _ in onset_bias_pairs}
    ) != len(onset_bias_pairs):
        raise ValueError(
            "lane checkpoint response repair onset rows must be sorted and unique"
        )
    if not hasattr(model, "set_response_token_repair_onset_row_biases"):
        if onset_bias_pairs:
            raise ValueError(
                "response repair onset row biases require a micro action model"
            )
    else:
        model.set_response_token_repair_onset_row_biases(onset_bias_pairs)
    onset_scope_value = metadata.get(
        "response_token_repair_scope_until_first_semantic_token", "false"
    ).lower()
    if onset_scope_value not in {"true", "false"}:
        raise ValueError(
            "lane checkpoint has an invalid response repair onset-scope flag"
        )
    if not hasattr(
        model,
        "set_response_token_repair_scope_until_first_semantic_token",
    ):
        if onset_scope_value == "true":
            raise ValueError(
                "response repair onset scope requires a micro action model"
            )
    else:
        model.set_response_token_repair_scope_until_first_semantic_token(
            onset_scope_value == "true"
        )
    post_onset_rows_text = metadata.get(
        "response_token_repair_post_onset_disabled_rows", ""
    )
    try:
        post_onset_rows = tuple(
            int(value) for value in post_onset_rows_text.split(",") if value
        )
    except ValueError as error:
        raise ValueError(
            "lane checkpoint has invalid post-onset disabled repair rows"
        ) from error
    if ",".join(str(row_id) for row_id in post_onset_rows) != post_onset_rows_text:
        raise ValueError(
            "lane checkpoint post-onset disabled repair rows must be canonical"
        )
    if tuple(sorted(post_onset_rows)) != post_onset_rows or len(
        set(post_onset_rows)
    ) != len(post_onset_rows):
        raise ValueError(
            "lane checkpoint post-onset disabled repair rows must be sorted and unique"
        )
    if not hasattr(model, "set_response_token_repair_post_onset_disabled_rows"):
        if post_onset_rows:
            raise ValueError(
                "post-onset disabled repair rows require a micro action model"
            )
    else:
        model.set_response_token_repair_post_onset_disabled_rows(
            post_onset_rows
        )
    post_onset_bias_text = metadata.get(
        "response_token_repair_post_onset_row_biases", ""
    )
    post_onset_bias_pairs: list[tuple[int, float]] = []
    try:
        for item in post_onset_bias_text.split(","):
            if not item:
                continue
            row_text, bias_text = item.split(":", maxsplit=1)
            post_onset_bias_pairs.append((int(row_text), float(bias_text)))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "lane checkpoint has invalid post-onset repair row biases"
        ) from error
    canonical_post_onset_bias_text = ",".join(
        f"{row_id}:{format(bias, '.17g')}"
        for row_id, bias in post_onset_bias_pairs
    )
    if canonical_post_onset_bias_text != post_onset_bias_text:
        raise ValueError(
            "lane checkpoint post-onset repair row biases must be canonical"
        )
    if tuple(sorted(post_onset_bias_pairs)) != tuple(post_onset_bias_pairs) or len(
        {row_id for row_id, _ in post_onset_bias_pairs}
    ) != len(post_onset_bias_pairs):
        raise ValueError(
            "lane checkpoint post-onset repair bias rows must be sorted and unique"
        )
    if not hasattr(model, "set_response_token_repair_post_onset_row_biases"):
        if post_onset_bias_pairs:
            raise ValueError(
                "post-onset repair row biases require a micro action model"
            )
    else:
        model.set_response_token_repair_post_onset_row_biases(
            post_onset_bias_pairs
        )
    onset_latch_value = metadata.get(
        "response_token_repair_onset_generated_semantic_latch", "false"
    ).lower()
    if onset_latch_value not in {"true", "false"}:
        raise ValueError(
            "lane checkpoint has an invalid response repair onset-latch flag"
        )
    if not hasattr(
        model,
        "set_response_token_repair_onset_generated_semantic_latch",
    ):
        if onset_latch_value == "true":
            raise ValueError(
                "response repair onset latch requires a micro action model"
            )
    else:
        model.set_response_token_repair_onset_generated_semantic_latch(
            onset_latch_value == "true"
        )
    eos_response_value = metadata.get(
        "response_eos_conditioning", "false"
    ).lower()
    if eos_response_value not in {"true", "false"}:
        raise ValueError("lane checkpoint has an invalid response-EOS flag")
    if eos_response_value == "true":
        if not hasattr(model, "response_eos_conditioning_enabled"):
            raise ValueError("response EOS requires a micro action model")
        required_eos = {
            name for name in model_state if name.startswith("response_eos_head.")
        }
        provided_eos = provided & required_eos
        if not required_eos or provided_eos != required_eos:
            missing_eos = sorted(required_eos - provided_eos)
            raise ValueError(
                "enabled response EOS is incomplete; "
                f"missing={missing_eos[:3]}"
            )
        if not model.response_temporal_conditioning_enabled:
            raise ValueError("response EOS requires temporal response conditioning")
        model.response_eos_conditioning_enabled = True
    elif hasattr(model, "response_eos_conditioning_enabled"):
        model.response_eos_conditioning_enabled = False
    return metadata


__all__ = [
    "AGENT_LANE_CHECKPOINT_FORMAT",
    "agent_lane_state_dict",
    "load_agent_lane_checkpoint",
    "save_agent_lane_checkpoint",
]
