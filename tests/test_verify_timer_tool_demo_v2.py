from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_timer_tool_demo_v2 import timer_spoken_grounding_evidence  # noqa: E402


def test_duration_unit_is_semantic_timer_evidence_for_exact_timer_trace() -> None:
    evidence = timer_spoken_grounding_evidence("Sure, set a 2 minutes.")

    assert evidence["passed"] is True
    assert evidence["evidence_mode"] == "duration_unit"
    assert evidence["timer_literal_present"] is False
    assert evidence["minute_unit_present"] is True


def test_literal_timer_word_remains_the_stronger_quality_evidence() -> None:
    evidence = timer_spoken_grounding_evidence("Timer started for 2 minutes.")

    assert evidence["passed"] is True
    assert evidence["evidence_mode"] == "literal_timer"
    assert evidence["timer_literal_present"] is True


def test_duration_without_action_does_not_pass() -> None:
    evidence = timer_spoken_grounding_evidence("2 minutes.")

    assert evidence["passed"] is False
    assert evidence["action_present"] is False


def test_action_without_duration_does_not_pass() -> None:
    evidence = timer_spoken_grounding_evidence("The timer is set.")

    assert evidence["passed"] is False
    assert evidence["duration_present"] is False


def test_number_substrings_do_not_fake_the_two_minute_duration() -> None:
    evidence = timer_spoken_grounding_evidence("The timer is set for 20 minutes.")

    assert evidence["passed"] is False
