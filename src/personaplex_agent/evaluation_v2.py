"""Dependency-free held-out scoring for PersonaPlex-Agent V2 inference.

GPU inference is responsible only for producing the JSON-safe trace contract
defined here: completed action packets, terminal observation positions, and
assistant result-claim spans.  This module reparses packets with the fixed V2
grammar and computes deterministic tool/ref/argument, no-tool, and causal
result-grounding metrics entirely on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .action_v2 import (
    ACTION_V2_VERSION,
    DEFAULT_MANIFEST,
    PACKET_FRAMES,
    ActionKind,
    ActionPacketV2,
    ActionParser,
    ActionV2Manifest,
    ParsedAction,
)


EVALUATION_V2_VERSION = "ppx-heldout-eval-v2.0.0"


class EvaluationV2Error(ValueError):
    """Raised when an expectation or inference trace violates the contract."""


def _require_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluationV2Error(f"{name} must be a non-empty string")
    return value


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvaluationV2Error(f"{name} must be an integer >= {minimum}")
    return value


def _normalise_ref(ref: int | str) -> int:
    if isinstance(ref, bool):
        raise EvaluationV2Error("REF must be REF_0..REF_3 or an integer index")
    if isinstance(ref, str) and ref.startswith("REF_"):
        try:
            ref = int(ref.removeprefix("REF_"))
        except ValueError as exc:
            raise EvaluationV2Error("REF must be REF_0..REF_3") from exc
    if not isinstance(ref, int) or ref < 0 or ref >= DEFAULT_MANIFEST.max_refs:
        raise EvaluationV2Error("REF must be REF_0..REF_3")
    return ref


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


@dataclass(frozen=True, slots=True)
class ExpectedCallV2:
    """One exact held-out tool-call target in chronological order."""

    tool_name: str
    ref: int | str
    arguments: Mapping[str, str]

    def __post_init__(self) -> None:
        tool_name = _require_string("expected tool_name", self.tool_name)
        try:
            tool = DEFAULT_MANIFEST.tool(tool_name)
        except ValueError as exc:
            raise EvaluationV2Error(str(exc)) from exc
        ref = _normalise_ref(self.ref)
        arguments = dict(self.arguments)
        expected_names = tuple(argument.name for argument in tool.arguments)
        if set(arguments) != set(expected_names) or len(arguments) != len(expected_names):
            raise EvaluationV2Error(
                f"expected arguments for {tool_name!r} must be exactly {expected_names}"
            )
        for argument in tool.arguments:
            value = arguments[argument.name]
            if not isinstance(value, str) or value not in argument.allowed_tokens:
                raise EvaluationV2Error(
                    f"expected argument {argument.name!r} is outside its fixed token domain"
                )
        ordered = {name: arguments[name] for name in expected_names}
        object.__setattr__(self, "ref", ref)
        object.__setattr__(self, "arguments", MappingProxyType(ordered))

    def matches(self, action: ParsedAction) -> bool:
        return (
            action.kind is ActionKind.CALL
            and action.tool_name == self.tool_name
            and action.ref == self.ref
            and dict(action.arguments) == dict(self.arguments)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "tool_name": self.tool_name,
            "ref": self.ref,
            "arguments": dict(self.arguments),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExpectedCallV2":
        try:
            return cls(value["tool_name"], value["ref"], value["arguments"])
        except (KeyError, TypeError) as exc:
            raise EvaluationV2Error("invalid expected-call record") from exc


@dataclass(frozen=True, slots=True)
class ExpectedGroundingV2:
    """Ground truth for one terminal event and its first visible frame."""

    event_id: str
    ref: int | str
    visible_frame: int
    claim_required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _require_string("grounding event_id", self.event_id))
        object.__setattr__(self, "ref", _normalise_ref(self.ref))
        object.__setattr__(
            self,
            "visible_frame",
            _require_int("grounding visible_frame", self.visible_frame),
        )
        if not isinstance(self.claim_required, bool):
            raise EvaluationV2Error("claim_required must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "ref": self.ref,
            "visible_frame": self.visible_frame,
            "claim_required": self.claim_required,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExpectedGroundingV2":
        try:
            return cls(
                value["event_id"],
                value["ref"],
                value["visible_frame"],
                value.get("claim_required", True),
            )
        except (KeyError, TypeError) as exc:
            raise EvaluationV2Error("invalid expected-grounding record") from exc


@dataclass(frozen=True, slots=True)
class HeldOutCaseV2:
    """One held-out expectation, independent of a particular GPU runner."""

    case_id: str
    expect_no_tool: bool
    expected_calls: tuple[ExpectedCallV2, ...] = ()
    expected_groundings: tuple[ExpectedGroundingV2, ...] = ()
    manifest_fingerprint: str = DEFAULT_MANIFEST.fingerprint

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _require_string("held-out case_id", self.case_id))
        if not isinstance(self.expect_no_tool, bool):
            raise EvaluationV2Error("expect_no_tool must be boolean")
        calls = tuple(self.expected_calls)
        groundings = tuple(self.expected_groundings)
        if not all(isinstance(call, ExpectedCallV2) for call in calls):
            raise EvaluationV2Error("expected_calls must contain ExpectedCallV2 values")
        if not all(isinstance(item, ExpectedGroundingV2) for item in groundings):
            raise EvaluationV2Error(
                "expected_groundings must contain ExpectedGroundingV2 values"
            )
        if self.expect_no_tool and calls:
            raise EvaluationV2Error("a no-tool case cannot require tool calls")
        if not self.expect_no_tool and not calls:
            raise EvaluationV2Error("a tool case must require at least one exact call")
        event_ids = [item.event_id for item in groundings]
        if len(event_ids) != len(set(event_ids)):
            raise EvaluationV2Error("expected grounding event IDs must be unique")
        if self.manifest_fingerprint != DEFAULT_MANIFEST.fingerprint:
            raise EvaluationV2Error("held-out case manifest fingerprint is not fixed V2")
        object.__setattr__(self, "expected_calls", calls)
        object.__setattr__(self, "expected_groundings", groundings)

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "expect_no_tool": self.expect_no_tool,
            "expected_calls": [call.to_dict() for call in self.expected_calls],
            "expected_groundings": [item.to_dict() for item in self.expected_groundings],
            "manifest_fingerprint": self.manifest_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HeldOutCaseV2":
        try:
            return cls(
                case_id=value["case_id"],
                expect_no_tool=value["expect_no_tool"],
                expected_calls=tuple(
                    ExpectedCallV2.from_dict(item) for item in value.get("expected_calls", ())
                ),
                expected_groundings=tuple(
                    ExpectedGroundingV2.from_dict(item)
                    for item in value.get("expected_groundings", ())
                ),
                manifest_fingerprint=value.get(
                    "manifest_fingerprint", DEFAULT_MANIFEST.fingerprint
                ),
            )
        except (KeyError, TypeError) as exc:
            raise EvaluationV2Error("invalid held-out case record") from exc


@dataclass(frozen=True, slots=True)
class TerminalEventTraceV2:
    """Terminal event timing emitted by the live/offline environment scheduler."""

    event_id: str
    ref: int | str
    injected_end_frame: int
    injected_end_micro: int
    visible_frame: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _require_string("terminal event_id", self.event_id))
        object.__setattr__(self, "ref", _normalise_ref(self.ref))
        end_frame = _require_int("terminal injected_end_frame", self.injected_end_frame)
        end_micro = _require_int("terminal injected_end_micro", self.injected_end_micro)
        if end_micro >= 5:
            raise EvaluationV2Error("terminal injected_end_micro must be 0..4")
        visible = _require_int("terminal visible_frame", self.visible_frame)
        if visible != end_frame + 1:
            raise EvaluationV2Error("terminal visible_frame must equal injected_end_frame + 1")

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "ref": self.ref,
            "injected_end_frame": self.injected_end_frame,
            "injected_end_micro": self.injected_end_micro,
            "visible_frame": self.visible_frame,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TerminalEventTraceV2":
        try:
            return cls(
                value["event_id"],
                value["ref"],
                value["injected_end_frame"],
                value["injected_end_micro"],
                value["visible_frame"],
            )
        except (KeyError, TypeError) as exc:
            raise EvaluationV2Error("invalid terminal-event trace") from exc


@dataclass(frozen=True, slots=True)
class GroundedClaimTraceV2:
    """A result-dependent assistant speech/text span identified by inference."""

    claim_id: str
    terminal_event_id: str
    ref: int | str
    start_frame: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _require_string("claim_id", self.claim_id))
        object.__setattr__(
            self,
            "terminal_event_id",
            _require_string("claim terminal_event_id", self.terminal_event_id),
        )
        object.__setattr__(self, "ref", _normalise_ref(self.ref))
        object.__setattr__(self, "start_frame", _require_int("claim start_frame", self.start_frame))

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "terminal_event_id": self.terminal_event_id,
            "ref": self.ref,
            "start_frame": self.start_frame,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GroundedClaimTraceV2":
        try:
            return cls(
                value["claim_id"],
                value["terminal_event_id"],
                value["ref"],
                value["start_frame"],
            )
        except (KeyError, TypeError) as exc:
            raise EvaluationV2Error("invalid grounded-claim trace") from exc


@dataclass(frozen=True, slots=True)
class InferenceTraceV2:
    """JSON-safe output contract for a future CPU or GPU inference runner."""

    case_id: str
    action_packets: tuple[ActionPacketV2, ...] = ()
    terminal_events: tuple[TerminalEventTraceV2, ...] = ()
    grounded_claims: tuple[GroundedClaimTraceV2, ...] = ()
    manifest_fingerprint: str = DEFAULT_MANIFEST.fingerprint

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _require_string("inference case_id", self.case_id))
        packets = tuple(self.action_packets)
        terminal_events = tuple(self.terminal_events)
        claims = tuple(self.grounded_claims)
        if not all(isinstance(packet, ActionPacketV2) for packet in packets):
            raise EvaluationV2Error("action_packets must contain ActionPacketV2 values")
        if not all(isinstance(item, TerminalEventTraceV2) for item in terminal_events):
            raise EvaluationV2Error("terminal_events must contain TerminalEventTraceV2 values")
        if not all(isinstance(item, GroundedClaimTraceV2) for item in claims):
            raise EvaluationV2Error("grounded_claims must contain GroundedClaimTraceV2 values")
        if len({item.event_id for item in terminal_events}) != len(terminal_events):
            raise EvaluationV2Error("terminal trace event IDs must be unique")
        if len({item.claim_id for item in claims}) != len(claims):
            raise EvaluationV2Error("grounded claim IDs must be unique")
        if self.manifest_fingerprint != DEFAULT_MANIFEST.fingerprint:
            raise EvaluationV2Error("inference trace manifest fingerprint is not fixed V2")
        object.__setattr__(self, "action_packets", packets)
        object.__setattr__(self, "terminal_events", terminal_events)
        object.__setattr__(self, "grounded_claims", claims)

    @classmethod
    def empty(cls, case_id: str) -> "InferenceTraceV2":
        return cls(case_id=case_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "action_packets": [
                {
                    "packet_start_frame": packet.packet_start_frame,
                    "token_ids": list(packet.token_ids),
                }
                for packet in self.action_packets
            ],
            "terminal_events": [item.to_dict() for item in self.terminal_events],
            "grounded_claims": [item.to_dict() for item in self.grounded_claims],
            "manifest_fingerprint": self.manifest_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InferenceTraceV2":
        try:
            return cls(
                case_id=value["case_id"],
                action_packets=tuple(
                    ActionPacketV2(item["packet_start_frame"], tuple(item["token_ids"]))
                    for item in value.get("action_packets", ())
                ),
                terminal_events=tuple(
                    TerminalEventTraceV2.from_dict(item)
                    for item in value.get("terminal_events", ())
                ),
                grounded_claims=tuple(
                    GroundedClaimTraceV2.from_dict(item)
                    for item in value.get("grounded_claims", ())
                ),
                manifest_fingerprint=value.get(
                    "manifest_fingerprint", DEFAULT_MANIFEST.fingerprint
                ),
            )
        except (KeyError, TypeError) as exc:
            raise EvaluationV2Error("invalid inference trace record") from exc


@dataclass(frozen=True, slots=True)
class CaseMetricsV2:
    valid_packet_count: int
    invalid_packet_count: int
    expected_call_count: int
    predicted_call_count: int
    exact_call_count: int
    tool_match_count: int
    ref_match_count: int
    argument_match_count: int
    false_positive_call_count: int
    missed_call_count: int
    no_tool_correct: bool | None
    terminal_trace_match_count: int
    terminal_trace_error_count: int
    grounding_claim_count: int
    valid_grounded_claim_count: int
    early_grounding_claim_count: int
    unknown_grounding_claim_count: int
    required_grounding_count: int
    grounded_required_event_count: int
    missing_grounding_count: int
    grounding_lag_sum_frames: int

    @property
    def mean_grounding_lag_frames(self) -> float | None:
        return _rate(self.grounding_lag_sum_frames, self.valid_grounded_claim_count)

    def to_dict(self) -> dict[str, object]:
        result = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }
        result["mean_grounding_lag_frames"] = self.mean_grounding_lag_frames
        return result


@dataclass(frozen=True, slots=True)
class CaseEvaluationV2:
    case_id: str
    action_pass: bool
    grounding_pass: bool
    passed: bool
    metrics: CaseMetricsV2
    parsed_calls: tuple[dict[str, object], ...]
    invalid_packets: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "action_pass": self.action_pass,
            "grounding_pass": self.grounding_pass,
            "passed": self.passed,
            "metrics": self.metrics.to_dict(),
            "parsed_calls": [dict(item) for item in self.parsed_calls],
            "invalid_packets": [dict(item) for item in self.invalid_packets],
        }


@dataclass(frozen=True, slots=True)
class EvaluationReportV2:
    cases: tuple[CaseEvaluationV2, ...]
    aggregate: Mapping[str, object]
    evaluation_version: str = EVALUATION_V2_VERSION
    protocol_version: str = ACTION_V2_VERSION
    manifest_fingerprint: str = DEFAULT_MANIFEST.fingerprint

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(self, "aggregate", MappingProxyType(dict(self.aggregate)))

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluation_version": self.evaluation_version,
            "protocol_version": self.protocol_version,
            "manifest_fingerprint": self.manifest_fingerprint,
            "aggregate": dict(self.aggregate),
            "cases": [case.to_dict() for case in self.cases],
        }

    def serialize_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")


class HeldOutV2Evaluator:
    """Reparse inference outputs and score exact actions plus causal traces."""

    def __init__(self, manifest: ActionV2Manifest = DEFAULT_MANIFEST) -> None:
        if manifest.fingerprint != DEFAULT_MANIFEST.fingerprint:
            raise EvaluationV2Error("evaluator requires the fixed V2 manifest")
        self.manifest = manifest
        self.parser = ActionParser(manifest)

    def score_case(
        self,
        expected: HeldOutCaseV2,
        predicted: InferenceTraceV2,
    ) -> CaseEvaluationV2:
        if expected.case_id != predicted.case_id:
            raise EvaluationV2Error("expected and predicted case IDs do not match")
        if expected.manifest_fingerprint != predicted.manifest_fingerprint:
            raise EvaluationV2Error("expected and predicted manifest fingerprints do not match")

        valid_count = 0
        invalid: list[dict[str, object]] = []
        parsed_calls: list[ParsedAction] = []
        occupied_frames: set[int] = set()
        ordered_packets = sorted(
            enumerate(predicted.action_packets),
            key=lambda item: (item[1].packet_start_frame, item[0]),
        )
        for original_index, packet in ordered_packets:
            physical = set(
                range(packet.packet_start_frame, packet.packet_start_frame + PACKET_FRAMES)
            )
            overlap = sorted(physical & occupied_frames)
            if overlap:
                invalid.append(
                    {
                        "packet_index": original_index,
                        "packet_start_frame": packet.packet_start_frame,
                        "error": f"overlapping logical packet frames {overlap}",
                    }
                )
                continue
            occupied_frames.update(physical)
            try:
                action = self.parser.parse(packet)
            except ValueError as exc:
                invalid.append(
                    {
                        "packet_index": original_index,
                        "packet_start_frame": packet.packet_start_frame,
                        "error": str(exc),
                    }
                )
                continue
            valid_count += 1
            if action.kind is ActionKind.CALL:
                parsed_calls.append(action)

        expected_calls = expected.expected_calls
        aligned = min(len(expected_calls), len(parsed_calls))
        tool_matches = sum(
            expected_calls[index].tool_name == parsed_calls[index].tool_name
            for index in range(aligned)
        )
        ref_matches = sum(
            expected_calls[index].ref == parsed_calls[index].ref
            for index in range(aligned)
        )
        argument_matches = sum(
            dict(expected_calls[index].arguments) == dict(parsed_calls[index].arguments)
            for index in range(aligned)
        )
        exact_matches = sum(
            expected_calls[index].matches(parsed_calls[index])
            for index in range(aligned)
        )
        no_tool_correct = (
            len(parsed_calls) == 0 and not invalid if expected.expect_no_tool else None
        )
        action_pass = (
            not invalid
            and len(parsed_calls) == len(expected_calls)
            and exact_matches == len(expected_calls)
            and (no_tool_correct is not False)
        )

        expected_events = {item.event_id: item for item in expected.expected_groundings}
        observed_events = {item.event_id: item for item in predicted.terminal_events}
        trace_valid: set[str] = set()
        trace_errors = 0
        for event_id, grounding in expected_events.items():
            observed = observed_events.get(event_id)
            if observed is None or observed.ref != grounding.ref or (
                observed.visible_frame != grounding.visible_frame
            ):
                trace_errors += 1
            else:
                trace_valid.add(event_id)
        trace_errors += len(set(observed_events) - set(expected_events))

        valid_claims = 0
        early_claims = 0
        unknown_claims = 0
        lag_sum = 0
        grounded_events: set[str] = set()
        for claim in predicted.grounded_claims:
            grounding = expected_events.get(claim.terminal_event_id)
            if (
                grounding is None
                or claim.ref != grounding.ref
                or claim.terminal_event_id not in trace_valid
            ):
                unknown_claims += 1
                continue
            if claim.start_frame < grounding.visible_frame:
                early_claims += 1
                continue
            valid_claims += 1
            lag_sum += claim.start_frame - grounding.visible_frame
            grounded_events.add(grounding.event_id)

        required_events = {
            item.event_id for item in expected.expected_groundings if item.claim_required
        }
        missing_events = required_events - grounded_events
        grounding_pass = (
            trace_errors == 0
            and early_claims == 0
            and unknown_claims == 0
            and not missing_events
        )
        metrics = CaseMetricsV2(
            valid_packet_count=valid_count,
            invalid_packet_count=len(invalid),
            expected_call_count=len(expected_calls),
            predicted_call_count=len(parsed_calls),
            exact_call_count=exact_matches,
            tool_match_count=tool_matches,
            ref_match_count=ref_matches,
            argument_match_count=argument_matches,
            false_positive_call_count=len(parsed_calls) - exact_matches,
            missed_call_count=len(expected_calls) - exact_matches,
            no_tool_correct=no_tool_correct,
            terminal_trace_match_count=len(trace_valid),
            terminal_trace_error_count=trace_errors,
            grounding_claim_count=len(predicted.grounded_claims),
            valid_grounded_claim_count=valid_claims,
            early_grounding_claim_count=early_claims,
            unknown_grounding_claim_count=unknown_claims,
            required_grounding_count=len(required_events),
            grounded_required_event_count=len(required_events & grounded_events),
            missing_grounding_count=len(missing_events),
            grounding_lag_sum_frames=lag_sum,
        )
        summaries = tuple(
            {
                "tool_name": action.tool_name,
                "ref": action.ref,
                "arguments": dict(action.arguments),
                "issued_frame": action.issued_frame,
                "issued_micro": action.issued_micro,
            }
            for action in parsed_calls
        )
        return CaseEvaluationV2(
            case_id=expected.case_id,
            action_pass=action_pass,
            grounding_pass=grounding_pass,
            passed=action_pass and grounding_pass,
            metrics=metrics,
            parsed_calls=summaries,
            invalid_packets=tuple(invalid),
        )

    def score_suite(
        self,
        expectations: Iterable[HeldOutCaseV2],
        predictions: Iterable[InferenceTraceV2],
    ) -> EvaluationReportV2:
        expected_values = tuple(expectations)
        predicted_values = tuple(predictions)
        expected_by_id = {item.case_id: item for item in expected_values}
        predicted_by_id = {item.case_id: item for item in predicted_values}
        if len(expected_by_id) != len(expected_values):
            raise EvaluationV2Error("held-out case IDs must be unique")
        if len(predicted_by_id) != len(predicted_values):
            raise EvaluationV2Error("inference trace case IDs must be unique")
        extras = set(predicted_by_id) - set(expected_by_id)
        if extras:
            raise EvaluationV2Error(f"predictions contain unknown held-out cases: {sorted(extras)}")
        missing = set(expected_by_id) - set(predicted_by_id)
        if missing:
            raise EvaluationV2Error(f"predictions are missing held-out cases: {sorted(missing)}")
        scored = tuple(
            self.score_case(
                expected,
                predicted_by_id[case_id],
            )
            for case_id, expected in expected_by_id.items()
        )
        return EvaluationReportV2(scored, self._aggregate(scored))

    def score_records(
        self,
        expectation_records: Iterable[Mapping[str, Any]],
        prediction_records: Iterable[Mapping[str, Any]],
    ) -> EvaluationReportV2:
        """Convenience bridge for JSON/JSONL GPU runner output records."""

        return self.score_suite(
            (HeldOutCaseV2.from_dict(item) for item in expectation_records),
            (InferenceTraceV2.from_dict(item) for item in prediction_records),
        )

    @staticmethod
    def _aggregate(cases: tuple[CaseEvaluationV2, ...]) -> dict[str, object]:
        metrics = [case.metrics for case in cases]
        total_packets = sum(item.valid_packet_count + item.invalid_packet_count for item in metrics)
        valid_packets = sum(item.valid_packet_count for item in metrics)
        expected_calls = sum(item.expected_call_count for item in metrics)
        predicted_calls = sum(item.predicted_call_count for item in metrics)
        exact_calls = sum(item.exact_call_count for item in metrics)
        aligned_calls = sum(
            min(item.expected_call_count, item.predicted_call_count) for item in metrics
        )
        no_tool = [item for item in metrics if item.no_tool_correct is not None]
        claims = sum(item.grounding_claim_count for item in metrics)
        valid_claims = sum(item.valid_grounded_claim_count for item in metrics)
        required_groundings = sum(item.required_grounding_count for item in metrics)
        grounded_events = sum(item.grounded_required_event_count for item in metrics)
        lag_sum = sum(item.grounding_lag_sum_frames for item in metrics)
        return {
            "case_count": len(cases),
            "passed_case_count": sum(case.passed for case in cases),
            "case_pass_rate": _rate(sum(case.passed for case in cases), len(cases)),
            "action_pass_rate": _rate(sum(case.action_pass for case in cases), len(cases)),
            "grounding_pass_rate": _rate(sum(case.grounding_pass for case in cases), len(cases)),
            "grammar_valid_packet_rate": _rate(valid_packets, total_packets),
            "expected_call_count": expected_calls,
            "predicted_call_count": predicted_calls,
            "exact_call_count": exact_calls,
            "exact_call_precision": _rate(exact_calls, predicted_calls),
            "exact_call_recall": _rate(exact_calls, expected_calls),
            "tool_accuracy_aligned": _rate(sum(item.tool_match_count for item in metrics), aligned_calls),
            "ref_accuracy_aligned": _rate(sum(item.ref_match_count for item in metrics), aligned_calls),
            "argument_accuracy_aligned": _rate(
                sum(item.argument_match_count for item in metrics), aligned_calls
            ),
            "no_tool_case_count": len(no_tool),
            "no_tool_accuracy": _rate(sum(item.no_tool_correct is True for item in no_tool), len(no_tool)),
            "terminal_trace_error_count": sum(item.terminal_trace_error_count for item in metrics),
            "result_claim_count": claims,
            "causal_result_claim_precision": _rate(valid_claims, claims),
            "early_result_claim_count": sum(item.early_grounding_claim_count for item in metrics),
            "unknown_result_claim_count": sum(item.unknown_grounding_claim_count for item in metrics),
            "required_grounding_count": required_groundings,
            "result_grounding_recall": _rate(grounded_events, required_groundings),
            "mean_grounding_lag_frames": _rate(lag_sum, valid_claims),
        }


__all__ = [
    "EVALUATION_V2_VERSION",
    "CaseEvaluationV2",
    "CaseMetricsV2",
    "EvaluationReportV2",
    "EvaluationV2Error",
    "ExpectedCallV2",
    "ExpectedGroundingV2",
    "GroundedClaimTraceV2",
    "HeldOutCaseV2",
    "HeldOutV2Evaluator",
    "InferenceTraceV2",
    "TerminalEventTraceV2",
]
