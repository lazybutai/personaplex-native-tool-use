from __future__ import annotations

from dataclasses import replace
import unittest

from personaplex_agent.action_v2 import (
    ActionCompiler,
    ActionKind,
    ActionParser,
    RefLeaseError,
)
from personaplex_agent.environment_v2 import EnvironmentEventV2, EnvironmentKind
from personaplex_agent.live_runtime_v2 import DuplexToolRuntimeV2, LiveRuntimeV2Error


class LiveRuntimeV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.compiler = ActionCompiler()
        self.parser = ActionParser()
        self.runtime = DuplexToolRuntimeV2(uuid_factory=lambda: "live-call-uuid")

    def _weather_call(self, start: int = 0, ref: int = 0):
        return self.parser.parse(
            self.compiler.compile_call(
                start,
                ref,
                "weather.lookup",
                {"city": "CITY_0"},
                active_refs=self.runtime.leases,
            ),
            active_refs=self.runtime.leases,
        )

    def test_terminal_ref_releases_only_after_live_obs_end_is_model_visible(self) -> None:
        dispatch = self.runtime.dispatch_action(self._weather_call())
        assert dispatch.lease is not None
        lease = dispatch.lease
        self.runtime.submit_event(
            EnvironmentEventV2(
                event_id="pending",
                event_seq=0,
                available_frame=1,
                issued_frame=lease.issued_frame,
                ref=lease.ref,
                tool_name="weather.lookup",
                kind=EnvironmentKind.PENDING,
                call_uuid=lease.call_uuid,
                lease=lease.lease,
            )
        )
        self.runtime.submit_event(
            EnvironmentEventV2(
                event_id="terminal",
                event_seq=1,
                available_frame=2,
                issued_frame=lease.issued_frame,
                ref=lease.ref,
                tool_name="weather.lookup",
                kind=EnvironmentKind.OK,
                payload_tokens=("FORECAST_00", "TEMP_C_20"),
                call_uuid=lease.call_uuid,
                lease=lease.lease,
            )
        )

        self.assertFalse(self.runtime.environment_frame(0).has_observation)
        self.assertTrue(self.runtime.environment_frame(1).has_observation)
        self.assertTrue(self.runtime.environment_frame(2).has_observation)
        terminal_frame = self.runtime.environment_frame(3)
        terminal = terminal_frame.completed_events[0]
        self.assertEqual(terminal.event.event_id, "terminal")
        self.assertEqual(terminal.visible_frame, 4)
        self.assertEqual(self.runtime.acknowledge_model_frame(3), ())
        self.assertEqual(self.runtime.active_refs, frozenset({0}))
        released = self.runtime.acknowledge_model_frame(4)
        self.assertEqual(tuple(item.ref for item in released), (0,))
        self.assertFalse(self.runtime.active_refs)

    def test_forged_issue_time_and_post_terminal_event_are_rejected(self) -> None:
        dispatch = self.runtime.dispatch_action(self._weather_call())
        assert dispatch.lease is not None
        lease = dispatch.lease
        with self.assertRaisesRegex(RefLeaseError, "issued_frame"):
            self.runtime.submit_event(
                EnvironmentEventV2(
                    event_id="forged",
                    event_seq=0,
                    available_frame=2,
                    issued_frame=1,
                    ref=lease.ref,
                    tool_name="weather.lookup",
                    kind=EnvironmentKind.ERROR,
                    payload_tokens=("ERR_EXECUTOR",),
                    call_uuid=lease.call_uuid,
                    lease=lease.lease,
                )
            )
        terminal = EnvironmentEventV2(
            event_id="terminal",
            event_seq=1,
            available_frame=2,
            issued_frame=lease.issued_frame,
            ref=lease.ref,
            tool_name="weather.lookup",
            kind=EnvironmentKind.ERROR,
            payload_tokens=("ERR_EXECUTOR",),
            call_uuid=lease.call_uuid,
            lease=lease.lease,
        )
        self.runtime.submit_event(terminal)
        with self.assertRaisesRegex(LiveRuntimeV2Error, "after.*terminal"):
            self.runtime.submit_event(
                EnvironmentEventV2(
                    event_id="late-pending",
                    event_seq=2,
                    available_frame=3,
                    issued_frame=lease.issued_frame,
                    ref=lease.ref,
                    tool_name="weather.lookup",
                    kind=EnvironmentKind.PENDING,
                    call_uuid=lease.call_uuid,
                    lease=lease.lease,
                )
            )

    def test_event_submitted_late_appears_now_never_in_a_past_frame(self) -> None:
        dispatch = self.runtime.dispatch_action(self._weather_call())
        assert dispatch.lease is not None
        lease = dispatch.lease
        for frame in range(5):
            self.assertFalse(self.runtime.environment_frame(frame).has_observation)
        self.runtime.submit_event(
            EnvironmentEventV2(
                event_id="late-delivery",
                event_seq=0,
                available_frame=1,
                issued_frame=lease.issued_frame,
                ref=lease.ref,
                tool_name="weather.lookup",
                kind=EnvironmentKind.TIMEOUT,
                payload_tokens=("ERR_EXECUTOR",),
                call_uuid=lease.call_uuid,
                lease=lease.lease,
            )
        )
        injected = self.runtime.environment_frame(5)
        self.assertTrue(injected.has_observation)
        self.assertFalse(injected.completed_events)
        completed = self.runtime.environment_frame(6).completed_events[0]
        self.assertEqual(completed.injected_start.frame, 5)

    def test_cancel_uses_the_existing_lease_and_controls_never_invoke_executor(self) -> None:
        call = self.runtime.dispatch_action(self._weather_call())
        assert call.lease is not None
        cancel = self.parser.parse(
            self.compiler.compile_cancel(2, 0, active_refs=self.runtime.leases),
            active_refs=self.runtime.leases,
        )
        cancel_dispatch = self.runtime.dispatch_action(cancel)
        self.assertEqual(cancel_dispatch.lease, call.lease)
        self.assertTrue(cancel_dispatch.executor_required)

        control = self.parser.parse(self.compiler.compile_control(4, "BACKCHANNEL"))
        control_dispatch = self.runtime.dispatch_action(control)
        self.assertIs(control.kind, ActionKind.CONTROL)
        self.assertIsNone(control_dispatch.lease)
        self.assertFalse(control_dispatch.executor_required)

    def test_live_clock_is_strictly_monotonic(self) -> None:
        self.runtime.environment_frame(0)
        with self.assertRaisesRegex(LiveRuntimeV2Error, "expected frame 1"):
            self.runtime.environment_frame(2)

    def test_dispatch_reparses_packet_and_rejects_forged_metadata_or_uuid_reuse(self) -> None:
        parsed = self._weather_call()
        forged = replace(parsed, tool_name="timer.create")
        with self.assertRaisesRegex(LiveRuntimeV2Error, "does not match"):
            self.runtime.dispatch_action(forged)

        self.runtime.dispatch_action(parsed, call_uuid="unique-id")
        second = self.parser.parse(
            self.compiler.compile_call(
                2,
                1,
                "timer.create",
                {"minutes": "MIN_1"},
                active_refs=self.runtime.leases,
            ),
            active_refs=self.runtime.leases,
        )
        with self.assertRaisesRegex(LiveRuntimeV2Error, "UUIDs must be unique"):
            self.runtime.dispatch_action(second, call_uuid="unique-id")


if __name__ == "__main__":
    unittest.main()
