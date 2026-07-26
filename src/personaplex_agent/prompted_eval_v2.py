"""Metrics for prompted native-action feature evaluation.

The cached evaluator is deliberately stricter than aggregate token accuracy.
Most supervised action positions are ``NOOP``, so readiness is decided from
whole tool packets, call timing, no-tool false positives, and repetitions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np

from .action_v2 import DEFAULT_MANIFEST, PAD_TOKEN_ID


def scenario_intent(scenario_id: str, *, expected_has_call: bool) -> str:
    """Map a catalog scenario ID to a stable evaluation family."""

    normalized = scenario_id.upper()
    for label in ("NO-TOOL", "WEATHER", "TIMER", "HOME"):
        if label in normalized:
            return label.lower()
    return "unknown" if expected_has_call else "no-tool"


def score_prompted_action_frames(
    expected: np.ndarray,
    predicted: np.ndarray,
    supervision: np.ndarray,
    *,
    call_start_frame: int | None,
    user_end_frame: int,
) -> dict[str, object]:
    """Score one free micro-decoder trace against its prompted target.

    Backbone states can be cached, but every predicted five-slot frame must be
    decoded autoregressively and grammar-constrained before it reaches here.
    """

    expected = np.asarray(expected)
    predicted = np.asarray(predicted)
    supervision = np.asarray(supervision, dtype=np.bool_)
    if expected.ndim != 2 or expected.shape[1] != 5:
        raise ValueError("expected prompted actions must be [T,5]")
    if predicted.shape != expected.shape or supervision.shape != expected.shape:
        raise ValueError("predicted actions and supervision must match [T,5]")
    if np.any(supervision & (expected == PAD_TOKEN_ID)):
        raise ValueError("supervision cannot select PAD targets")
    if not 0 <= user_end_frame <= expected.shape[0]:
        raise ValueError("user_end_frame is outside the action trace")
    if call_start_frame is not None and not 0 <= call_start_frame < expected.shape[0]:
        raise ValueError("call_start_frame is outside the action trace")

    noop_id = DEFAULT_MANIFEST.action_id("NOOP")
    call_begin_id = DEFAULT_MANIFEST.action_id("CALL_BEGIN")
    selected = supervision & (expected >= 0)
    noop_selected = selected & (expected == noop_id)
    non_noop_selected = selected & (expected != noop_id)
    tool_ids = {
        DEFAULT_MANIFEST.action_id(tool.action_token)
        for tool in DEFAULT_MANIFEST.tool_schemas
    }
    ref_ids = {
        DEFAULT_MANIFEST.action_id(f"REF_{ref}")
        for ref in range(DEFAULT_MANIFEST.max_refs)
    }
    argument_ids = {
        DEFAULT_MANIFEST.action_id(token)
        for tool in DEFAULT_MANIFEST.tool_schemas
        for argument in tool.arguments
        for token in argument.allowed_tokens
    }
    call_end_id = DEFAULT_MANIFEST.action_id("CALL_END")
    predicted_call_frames = tuple(
        int(frame)
        for frame in np.flatnonzero(predicted[:, 0] == call_begin_id).tolist()
    )
    predicted_call_rows = [
        {
            "frame": frame,
            "ids": [int(token) for token in predicted[frame].tolist()],
            "tokens": [
                "PAD" if int(token) < 0 else DEFAULT_MANIFEST.action_name(int(token))
                for token in predicted[frame].tolist()
            ],
        }
        for frame in predicted_call_frames
    ]
    expected_has_call = call_start_frame is not None
    expected_call_rows = np.flatnonzero(np.any(expected > noop_id, axis=1))
    exact_packet = False
    if expected_has_call:
        exact_packet = (
            len(predicted_call_frames) == 1
            and predicted_call_frames[0] == call_start_frame
            and bool(expected_call_rows.size)
            and np.array_equal(
                predicted[expected_call_rows], expected[expected_call_rows]
            )
        )

    def _accuracy(mask: np.ndarray) -> tuple[int, int]:
        total = int(mask.sum())
        correct = int(((predicted == expected) & mask).sum())
        return correct, total

    selected_correct, selected_total = _accuracy(selected)
    noop_correct, noop_total = _accuracy(noop_selected)
    non_noop_correct, non_noop_total = _accuracy(non_noop_selected)
    tool_mask = selected & np.isin(expected, tuple(tool_ids))
    ref_mask = selected & np.isin(expected, tuple(ref_ids))
    argument_mask = selected & np.isin(expected, tuple(argument_ids))
    terminator_mask = selected & (expected == call_end_id)
    tool_correct, tool_total = _accuracy(tool_mask)
    ref_correct, ref_total = _accuracy(ref_mask)
    argument_correct, argument_total = _accuracy(argument_mask)
    terminator_correct, terminator_total = _accuracy(terminator_mask)
    timing_error = None
    if expected_has_call and predicted_call_frames:
        timing_error = min(abs(frame - call_start_frame) for frame in predicted_call_frames)

    return {
        "expected_has_call": expected_has_call,
        "predicted_call_frames": list(predicted_call_frames),
        "predicted_call_rows": predicted_call_rows,
        "predicted_call_count": len(predicted_call_frames),
        "call_detected": bool(predicted_call_frames),
        "call_timing_exact": bool(
            expected_has_call and call_start_frame in predicted_call_frames
        ),
        "call_timing_error_frames": timing_error,
        "exact_call_packet": bool(exact_packet),
        "false_call": bool(not expected_has_call and predicted_call_frames),
        "premature_call": any(frame < user_end_frame for frame in predicted_call_frames),
        "repeated_call": len(predicted_call_frames) > (1 if expected_has_call else 0),
        "selected_correct": selected_correct,
        "selected_total": selected_total,
        "noop_correct": noop_correct,
        "noop_total": noop_total,
        "non_noop_correct": non_noop_correct,
        "non_noop_total": non_noop_total,
        "tool_correct": tool_correct,
        "tool_total": tool_total,
        "ref_correct": ref_correct,
        "ref_total": ref_total,
        "argument_correct": argument_correct,
        "argument_total": argument_total,
        "terminator_correct": terminator_correct,
        "terminator_total": terminator_total,
    }


def aggregate_prompted_scores(
    cases: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate case scores without allowing NOOPs to hide call failures."""

    rows = list(cases)
    if not rows:
        raise ValueError("at least one prompted evaluation case is required")

    def _rate(numerator: int, denominator: int) -> float | None:
        return None if denominator == 0 else round(numerator / denominator, 6)

    tool_rows = [row for row in rows if bool(row["expected_has_call"])]
    no_tool_rows = [row for row in rows if not bool(row["expected_has_call"])]
    selected_correct = sum(int(row["selected_correct"]) for row in rows)
    selected_total = sum(int(row["selected_total"]) for row in rows)
    noop_correct = sum(int(row["noop_correct"]) for row in rows)
    noop_total = sum(int(row["noop_total"]) for row in rows)
    non_noop_correct = sum(int(row["non_noop_correct"]) for row in rows)
    non_noop_total = sum(int(row["non_noop_total"]) for row in rows)
    tool_correct = sum(int(row["tool_correct"]) for row in rows)
    tool_total = sum(int(row["tool_total"]) for row in rows)
    ref_correct = sum(int(row["ref_correct"]) for row in rows)
    ref_total = sum(int(row["ref_total"]) for row in rows)
    argument_correct = sum(int(row["argument_correct"]) for row in rows)
    argument_total = sum(int(row["argument_total"]) for row in rows)
    terminator_correct = sum(int(row["terminator_correct"]) for row in rows)
    terminator_total = sum(int(row["terminator_total"]) for row in rows)
    scenario_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"cases": 0, "detected": 0, "exact_packet": 0, "false_call": 0}
    )
    intent_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "cases": 0,
            "expected_calls": 0,
            "detected": 0,
            "exact_packet": 0,
            "false_call": 0,
            "premature_call": 0,
            "repeated_call": 0,
        }
    )
    for row in rows:
        scenario = str(row.get("scenario_id", "unknown"))
        scenario_counts[scenario]["cases"] += 1
        scenario_counts[scenario]["detected"] += int(bool(row["call_detected"]))
        scenario_counts[scenario]["exact_packet"] += int(bool(row["exact_call_packet"]))
        scenario_counts[scenario]["false_call"] += int(bool(row["false_call"]))
        intent = scenario_intent(
            scenario,
            expected_has_call=bool(row["expected_has_call"]),
        )
        intent_counts[intent]["cases"] += 1
        intent_counts[intent]["expected_calls"] += int(bool(row["expected_has_call"]))
        intent_counts[intent]["detected"] += int(bool(row["call_detected"]))
        intent_counts[intent]["exact_packet"] += int(bool(row["exact_call_packet"]))
        intent_counts[intent]["false_call"] += int(bool(row["false_call"]))
        intent_counts[intent]["premature_call"] += int(bool(row["premature_call"]))
        intent_counts[intent]["repeated_call"] += int(bool(row["repeated_call"]))

    return {
        "feature_count": len(rows),
        "tool_feature_count": len(tool_rows),
        "no_tool_feature_count": len(no_tool_rows),
        "call_detection_rate": _rate(
            sum(int(bool(row["call_detected"])) for row in tool_rows), len(tool_rows)
        ),
        "exact_call_timing_rate": _rate(
            sum(int(bool(row["call_timing_exact"])) for row in tool_rows), len(tool_rows)
        ),
        "exact_call_packet_rate": _rate(
            sum(int(bool(row["exact_call_packet"])) for row in tool_rows), len(tool_rows)
        ),
        "no_tool_false_call_rate": _rate(
            sum(int(bool(row["false_call"])) for row in no_tool_rows), len(no_tool_rows)
        ),
        "premature_call_rate": _rate(
            sum(int(bool(row["premature_call"])) for row in rows), len(rows)
        ),
        "repeated_call_rate": _rate(
            sum(int(bool(row["repeated_call"])) for row in rows), len(rows)
        ),
        "selected_token_accuracy": _rate(selected_correct, selected_total),
        "noop_token_accuracy": _rate(noop_correct, noop_total),
        "non_noop_token_accuracy": _rate(non_noop_correct, non_noop_total),
        "tool_token_accuracy": _rate(tool_correct, tool_total),
        "ref_token_accuracy": _rate(ref_correct, ref_total),
        "argument_token_accuracy": _rate(argument_correct, argument_total),
        "terminator_token_accuracy": _rate(terminator_correct, terminator_total),
        "intent_counts": dict(sorted(intent_counts.items())),
        "scenario_counts": dict(sorted(scenario_counts.items())),
    }


def prompted_readiness_gate(
    aggregate: Mapping[str, object],
    *,
    min_call_detection_rate: float | None = None,
    min_exact_call_packet_rate: float | None = None,
    max_no_tool_false_call_rate: float | None = None,
    max_premature_call_rate: float | None = None,
    max_repeated_call_rate: float | None = None,
) -> dict[str, object]:
    """Apply explicit promotion thresholds to an aggregate evaluation."""

    thresholds = {
        "call_detection_rate": ("min", min_call_detection_rate),
        "exact_call_packet_rate": ("min", min_exact_call_packet_rate),
        "no_tool_false_call_rate": ("max", max_no_tool_false_call_rate),
        "premature_call_rate": ("max", max_premature_call_rate),
        "repeated_call_rate": ("max", max_repeated_call_rate),
    }
    failures: list[dict[str, object]] = []
    enabled: dict[str, dict[str, object]] = {}
    for metric, (direction, threshold) in thresholds.items():
        if threshold is None:
            continue
        if not 0 <= threshold <= 1:
            raise ValueError("readiness thresholds must be in [0,1]")
        observed = aggregate.get(metric)
        enabled[metric] = {"direction": direction, "threshold": threshold}
        failed = observed is None or (
            direction == "min" and float(observed) < threshold
        ) or (direction == "max" and float(observed) > threshold)
        if failed:
            failures.append(
                {
                    "metric": metric,
                    "direction": direction,
                    "threshold": threshold,
                    "observed": observed,
                }
            )
    return {
        "passed": not failures,
        "thresholds": enabled,
        "failures": failures,
    }


__all__ = [
    "aggregate_prompted_scores",
    "prompted_readiness_gate",
    "scenario_intent",
    "score_prompted_action_frames",
]
