from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from live_session_v2 import AcousticTurnWindow, TimerExecutor  # noqa: E402


class AcousticTurnWindowTests(unittest.TestCase):
    def test_leading_silence_does_not_open_call_window(self) -> None:
        vad = AcousticTurnWindow(
            threshold=0.01,
            min_speech_frames=2,
            end_silence_frames=2,
            call_window_frames=4,
        )
        for _ in range(20):
            self.assertIsNone(vad.update(0.0))
            vad.advance()
        self.assertFalse(vad.in_speech)
        self.assertFalse(vad.call_window_open)
        self.assertEqual(vad.utterance_count, 0)

    def test_completed_acoustic_turn_opens_and_expires_call_window(self) -> None:
        vad = AcousticTurnWindow(
            threshold=0.01,
            min_speech_frames=2,
            end_silence_frames=2,
            call_window_frames=4,
        )
        self.assertIsNone(vad.update(0.02))
        self.assertEqual(vad.update(0.02), "speech_start")
        self.assertTrue(vad.in_speech)
        self.assertIsNone(vad.update(0.0))
        self.assertEqual(vad.update(0.0), "speech_end")
        self.assertEqual(vad.utterance_count, 1)
        self.assertEqual(vad.call_window_remaining, 4)
        for remaining in (3, 2, 1, 0):
            vad.advance()
            self.assertEqual(vad.call_window_remaining, remaining)
        self.assertFalse(vad.call_window_open)

    def test_new_speech_closes_previous_call_window(self) -> None:
        vad = AcousticTurnWindow(
            threshold=0.01,
            min_speech_frames=2,
            end_silence_frames=1,
            call_window_frames=10,
        )
        vad.update(0.02)
        vad.update(0.02)
        self.assertEqual(vad.update(0.0), "speech_end")
        self.assertTrue(vad.call_window_open)
        vad.update(0.02)
        self.assertFalse(vad.call_window_open)

    def test_invalid_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AcousticTurnWindow(threshold=0.0)
        vad = AcousticTurnWindow()
        with self.assertRaises(ValueError):
            vad.update(float("nan"))
        with self.assertRaises(ValueError):
            vad.update(-0.1)


class TimerExecutorTests(unittest.TestCase):
    def test_rejects_non_allowlisted_tool_and_bad_duration(self) -> None:
        executor = TimerExecutor()
        unsupported = executor.execute("weather.lookup", {"minutes": "MIN_2"})
        malformed = executor.execute("timer.create", {"minutes": "2"})
        outside_range = executor.execute("timer.create", {"minutes": "MIN_0"})
        self.assertFalse(unsupported.ok)
        self.assertEqual(unsupported.payload_tokens, ("ERR_EXECUTOR",))
        self.assertFalse(malformed.ok)
        self.assertEqual(malformed.payload_tokens, ("ERR_SCHEMA",))
        self.assertFalse(outside_range.ok)

    def test_valid_timer_is_scheduled_without_blocking_session(self) -> None:
        executor = TimerExecutor()
        result = executor.execute("timer.create", {"minutes": "MIN_2"})
        self.assertTrue(result.ok)
        self.assertEqual(result.payload_tokens, ("ENUM_0",))
        self.assertEqual(result.scheduled_seconds, 120.0)
        self.assertEqual(len(executor._timers), 1)
        self.assertTrue(executor._timers[0].daemon)
        executor._timers[0].cancel()


if __name__ == "__main__":
    unittest.main()
