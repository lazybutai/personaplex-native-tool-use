"""Portable verifier for the promoted v169 persistent timer trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_SHA256 = "f9c5db254f8fea477155a9233b063b03248605c07e95efb987d88bc3626cdcba"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--proof-summary",
        action="store_true",
        help="Validate the portable proof-summary.json instead of a full live trace.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = json.loads(args.trace.read_text(encoding="utf-8"))
    proof = data["verified_proof"] if args.proof_summary else data

    if args.proof_summary:
        checks = {
            "format": data.get("format") == "personaplex-agent-lanes-v1",
            "checkpoint_sha": data.get("sha256") == EXPECTED_SHA256,
            "one_call": proof.get("call_count") == 1,
            "tool": proof.get("tool_name") == "timer.create",
            "arguments": proof.get("arguments") == {"minutes": "MIN_2"},
            "scheduled_seconds": proof.get("scheduled_seconds") == 120.0,
            "terminal": proof.get("terminal_result") == "OK",
            "causal_visibility": proof.get("result_visible_frame", -1)
            > proof.get("issued_frame", 10**9),
            "text": proof.get("generated_assistant_text") == "Sure, set a 2 minutes.",
            "audible": proof.get("assistant_audio_rms", 0.0) > 0.005,
        }
    else:
        calls = proof.get("calls", [])
        terminal = proof.get("terminal_results_visible_to_model", [])
        audio = proof.get("assistant_audio", {})
        checks = {
            "runner": proof.get("live_session_v2") == "ok",
            "one_call": len(calls) == 1,
            "tool": bool(calls and calls[0].get("tool_name") == "timer.create"),
            "arguments": bool(
                calls and calls[0].get("arguments") == {"minutes": "MIN_2"}
            ),
            "scheduled_seconds": bool(
                calls and float(calls[0].get("scheduled_seconds", -1)) == 120.0
            ),
            "one_terminal": len(terminal) == 1,
            "terminal_ok": bool(terminal and terminal[0].get("kind") == "OK"),
            "causal_visibility": bool(
                calls
                and terminal
                and int(terminal[0].get("visible_frame", -1))
                > int(calls[0].get("issued_frame", 10**9))
            ),
            "text": proof.get("generated_assistant_text") == "Sure, set a 2 minutes.",
            "audible": math.isfinite(float(audio.get("rms", 0.0)))
            and float(audio.get("rms", 0.0)) > 0.005,
        }

    if args.checkpoint is not None:
        checks["checkpoint_exists"] = args.checkpoint.is_file()
        checks["checkpoint_sha"] = (
            sha256(args.checkpoint) == EXPECTED_SHA256
            if args.checkpoint.is_file()
            else False
        )

    passed = all(checks.values())
    print(json.dumps({"passed": passed, "checks": checks}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
