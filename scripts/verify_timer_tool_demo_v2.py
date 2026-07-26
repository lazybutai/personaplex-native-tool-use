"""Verify the first exact held-out native TIMER call/result/audio trajectory."""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import re
import wave


VERIFIER_VERSION = "ppx-timer-tool-demo-verifier-v1.3.0"
EXPECTED_LANE_SHA256 = "2c603132cf2a74935309fe44635bd85db42429017f919e51a8df336f661bcbae"


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    root = _root()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace",
        type=Path,
        default=root / "artifacts" / "live-tool-v2" / "v70-timer003-sentence-latch.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / "artifacts"
        / "live-tool-v2"
        / "v70-timer003-sentence-latch-verification.json",
    )
    parser.add_argument("--expected-lane-sha256", default=EXPECTED_LANE_SHA256)
    parser.add_argument(
        "--expected-text-top-k",
        type=int,
        help="Require the recorded native text decoder top-k provenance.",
    )
    parser.add_argument(
        "--forbid-device",
        action="append",
        default=[],
        help="Reject a model or Mimi device name, for example cuda:2.",
    )
    parser.add_argument(
        "--require-spoken-grounding",
        action="store_true",
        help=(
            "Require the native reply to mention 2, a completed or active "
            "timer action, and either literal timer wording or minute-unit "
            "evidence in addition to the exact timer mechanics checks."
        ),
    )
    return parser.parse_args()


def timer_spoken_grounding_evidence(generated_text: str) -> dict[str, object]:
    """Score result grounding while preserving literal-word quality evidence."""

    tokens = tuple(re.findall(r"[a-z]+|\d+", generated_text.casefold()))
    duration_present = "2" in tokens
    action_present = any(token in {"set", "started", "running"} for token in tokens)
    timer_literal_present = "timer" in tokens
    minute_unit_present = any(token.startswith("minute") for token in tokens)
    semantic_timer_evidence = timer_literal_present or minute_unit_present
    passed = duration_present and action_present and semantic_timer_evidence
    return {
        "tokens": list(tokens),
        "duration_present": duration_present,
        "action_present": action_present,
        "timer_literal_present": timer_literal_present,
        "minute_unit_present": minute_unit_present,
        "semantic_timer_evidence": semantic_timer_evidence,
        "evidence_mode": (
            "literal_timer"
            if timer_literal_present
            else "duration_unit"
            if minute_unit_present
            else "none"
        ),
        "passed": passed,
    }


def main() -> int:
    args = parse_args()
    trace_path = args.trace.resolve()
    if not trace_path.is_file():
        raise FileNotFoundError(trace_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    calls = trace.get("calls", [])
    terminal = trace.get("terminal_results_visible_to_model", [])
    completions = trace.get("turn_completion_events", [])
    lane_path = Path(str(trace.get("lane_checkpoint", "")))
    audio_metadata = trace.get("assistant_audio", {})
    audio_path = Path(str(audio_metadata.get("path", "")))

    checks: dict[str, bool] = {}
    checks["exact_expected_call"] = trace.get("exact_expected_call") is True
    checks["one_call"] = len(calls) == 1
    checks["timer_packet"] = bool(
        len(calls) == 1
        and calls[0].get("tool_name") == "timer.create"
        and calls[0].get("arguments") == {"minutes": "MIN_2"}
    )
    checks["one_terminal_result"] = len(terminal) == 1
    visible_frame = int(terminal[0]["visible_frame"]) if terminal else 10**9
    checks["terminal_visible_after_call"] = bool(
        calls and visible_frame > int(calls[0]["issued_frame"])
    )
    checks["timer_result_injected"] = bool(
        calls and calls[0].get("mock_result_tokens") == ["ENUM_0"]
    )
    generated_text = str(trace.get("generated_assistant_text", "")).strip()
    checks["native_assistant_text_nonempty"] = bool(generated_text)
    checks["one_sentence_completion"] = bool(
        len(completions) == 1 and completions[0].get("kind") == "sentence_end"
    )
    completion_frame = int(completions[0]["frame"]) if completions else 10**9
    post_completion = [
        item
        for item in trace.get("text_sampling_trace", [])
        if int(item["frame"]) > completion_frame
    ]
    checks["post_completion_frames_present"] = len(post_completion) >= 16
    checks["post_completion_text_all_pad"] = bool(
        post_completion
        and all(int(item["emitted_token"]) == 3 for item in post_completion)
    )
    checks["pending_hold_present"] = bool(trace.get("pending_speech_frames"))

    if args.expected_text_top_k is not None:
        checks["text_top_k_exact"] = bool(
            trace.get("sampling", {}).get("text_top_k")
            == args.expected_text_top_k
        )
    forbidden_devices = {str(value).casefold() for value in args.forbid_device}
    if forbidden_devices:
        observed_devices = {
            str(trace.get("device", "")).casefold(),
            str(audio_metadata.get("device", "")).casefold(),
        }
        checks["forbidden_devices_absent"] = not bool(
            forbidden_devices & observed_devices
        )

    checks["lane_checkpoint_exists"] = lane_path.is_file()
    lane_sha = _sha256(lane_path) if lane_path.is_file() else None
    checks["lane_checkpoint_exact"] = lane_sha == args.expected_lane_sha256.lower()
    checks["assistant_wav_exists"] = audio_path.is_file()
    audio_rms = audio_peak = tail_rms = tail_peak = None
    if audio_path.is_file():
        with wave.open(str(audio_path), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            samples = array("h")
            samples.frombytes(handle.readframes(handle.getnframes()))
        checks["assistant_wav_pcm16_mono"] = channels == 1 and sample_width == 2
        normalized_audio = [value / 32768.0 for value in samples]
        if normalized_audio:
            audio_rms = math.sqrt(
                sum(value * value for value in normalized_audio)
                / len(normalized_audio)
            )
            audio_peak = max(abs(value) for value in normalized_audio)
        else:
            audio_rms = 0.0
            audio_peak = 0.0
        checks["assistant_wav_nonempty"] = bool(samples)
        checks["assistant_wav_audible"] = bool(
            audio_rms >= 0.001 and audio_peak >= 0.005
        )
        if completions:
            flush_seconds = (completion_frame + 8) / 12.5
            start = min(len(samples), int(flush_seconds * sample_rate) * channels)
            tail = samples[start:]
            normalized = [value / 32768.0 for value in tail]
        else:
            normalized = []
        if normalized:
            tail_rms = math.sqrt(
                sum(value * value for value in normalized) / len(normalized)
            )
            tail_peak = max(abs(value) for value in normalized)
        else:
            tail_rms = 0.0
            tail_peak = 0.0
        checks["post_flush_audio_silent"] = tail_rms <= 0.001 and tail_peak <= 0.005
    else:
        checks["assistant_wav_pcm16_mono"] = False
        checks["assistant_wav_nonempty"] = False
        checks["assistant_wav_audible"] = False
        checks["post_flush_audio_silent"] = False

    spoken_grounding = timer_spoken_grounding_evidence(generated_text)
    spoken_grounding_passed = bool(spoken_grounding["passed"])
    if args.require_spoken_grounding:
        checks["spoken_grounding"] = spoken_grounding_passed
    passed = all(checks.values())
    report = {
        "verify_timer_tool_demo_v2": "ok" if passed else "failed",
        "verifier_version": VERIFIER_VERSION,
        "scope": (
            "tool-call/result-visibility/native-audio/grounded-speech"
            if args.require_spoken_grounding
            else "tool-call/result-visibility/native-audio mechanics"
        ),
        "trace": str(trace_path),
        "trace_sha256": _sha256(trace_path),
        "lane_checkpoint_sha256": lane_sha,
        "visible_frame": visible_frame,
        "completion_frame": completion_frame,
        "post_completion_frame_count": len(post_completion),
        "assistant_audio_rms": audio_rms,
        "assistant_audio_peak": audio_peak,
        "post_flush_audio_rms": tail_rms,
        "post_flush_audio_peak": tail_peak,
        "generated_assistant_text": generated_text,
        "spoken_grounding_passed": spoken_grounding_passed,
        "spoken_grounding_evidence": spoken_grounding,
        "spoken_literal_timer_quality_passed": bool(
            spoken_grounding["timer_literal_present"]
        ),
        "spoken_grounding_note": (
            "Required semantically by this verification run; literal timer "
            "wording remains a separate quality flag."
            if args.require_spoken_grounding
            else "Informational only in mechanics mode."
        ),
        "checks": checks,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report["output"] = str(args.output.resolve())
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
