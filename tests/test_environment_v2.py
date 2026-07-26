from __future__ import annotations

import unittest

from personaplex_agent.action_v2 import ActionCompiler, ActionParser, DEFAULT_MANIFEST, RefLeaseError, RefLeaseGuard
from personaplex_agent.environment_v2 import (
    EnvironmentEventV2,
    EnvironmentFifoScheduler,
    EnvironmentFragmentError,
    EnvironmentKind,
    EnvironmentMicroLane,
    EnvironmentProtocolError,
    validate_environment_fragments,
)


class EnvironmentV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = DEFAULT_MANIFEST
        self.scheduler = EnvironmentFifoScheduler(self.manifest)

    def _reserve_timer_ref(self, ref: int = 0) -> tuple[RefLeaseGuard, object, object]:
        compiler = ActionCompiler(self.manifest)
        parser = ActionParser(self.manifest)
        packet = compiler.compile_call(
            4,
            ref,
            "timer.create",
            {"minutes": "MIN_3"},
        )
        parsed = parser.parse(packet)
        guard = RefLeaseGuard(self.manifest)
        lease = guard.reserve_parsed(parsed, call_uuid=f"hidden-call-{ref}")
        return guard, parsed, lease

    def _event(
        self,
        *,
        event_id: str,
        event_seq: int,
        available_frame: int,
        issued_frame: int,
        ref: int,
        kind: EnvironmentKind,
        payload_tokens: tuple[str, ...] = (),
        tool_name: str = "timer.create",
        call_uuid: str = "hidden-call-0",
        lease: int = 1,
    ) -> EnvironmentEventV2:
        return EnvironmentEventV2(
            event_id=event_id,
            event_seq=event_seq,
            available_frame=available_frame,
            issued_frame=issued_frame,
            ref=ref,
            tool_name=tool_name,
            kind=kind,
            payload_tokens=payload_tokens,
            call_uuid=call_uuid,
            lease=lease,
            manifest=self.manifest,
        )

    def test_event_record_requires_future_executor_provenance_and_fixed_payloads(self) -> None:
        with self.assertRaisesRegex(EnvironmentProtocolError, r"issued_frame \+ 1"):
            self._event(
                event_id="too-early",
                event_seq=0,
                available_frame=4,
                issued_frame=4,
                ref=0,
                kind=EnvironmentKind.PENDING,
            )
        with self.assertRaisesRegex(EnvironmentProtocolError, "PENDING"):
            self._event(
                event_id="pending-payload",
                event_seq=0,
                available_frame=5,
                issued_frame=4,
                ref=0,
                kind=EnvironmentKind.PENDING,
                payload_tokens=("ENUM_0",),
            )
        with self.assertRaisesRegex(EnvironmentProtocolError, "fixed result vocabulary"):
            self._event(
                event_id="wrong-ok-payload",
                event_seq=0,
                available_frame=5,
                issued_frame=4,
                ref=0,
                kind=EnvironmentKind.OK,
                payload_tokens=("OBS_END",),
            )
        with self.assertRaisesRegex(EnvironmentProtocolError, "tool_executor"):
            EnvironmentEventV2(
                event_id="wrong-source",
                event_seq=0,
                available_frame=5,
                issued_frame=4,
                ref=0,
                tool_name="timer.create",
                kind=EnvironmentKind.PENDING,
                call_uuid="hidden-call-0",
                lease=1,
                source="model",
                manifest=self.manifest,
            )

    def test_same_available_events_fifo_spill_and_terminal_release(self) -> None:
        guard, parsed, lease = self._reserve_timer_ref()
        pending = self._event(
            event_id="pending",
            event_seq=10,
            available_frame=parsed.issued_frame + 1,
            issued_frame=parsed.issued_frame,
            ref=0,
            kind=EnvironmentKind.PENDING,
            call_uuid=lease.call_uuid,
            lease=lease.lease,
        )
        result = self._event(
            event_id="result",
            event_seq=11,
            available_frame=parsed.issued_frame + 1,
            issued_frame=parsed.issued_frame,
            ref=0,
            kind=EnvironmentKind.OK,
            payload_tokens=("ENUM_0",),
            call_uuid=lease.call_uuid,
            lease=lease.lease,
        )

        schedule = self.scheduler.schedule(10, [result, pending], lease_guard=guard)
        scheduled_pending, scheduled_result = schedule.scheduled_events

        self.assertEqual([event.event.event_id for event in schedule.scheduled_events], ["pending", "result"])
        self.assertEqual(
            (scheduled_pending.injected_start.frame, scheduled_pending.injected_start.micro),
            (5, 0),
        )
        self.assertEqual(
            (scheduled_pending.injected_end.frame, scheduled_pending.injected_end.micro),
            (5, 4),
        )
        self.assertEqual(
            (scheduled_result.injected_start.frame, scheduled_result.injected_start.micro),
            (6, 0),
        )
        self.assertEqual(
            (scheduled_result.injected_end.frame, scheduled_result.injected_end.micro),
            (7, 2),
        )
        self.assertEqual(scheduled_result.visible_frame, scheduled_result.injected_end.frame + 1)
        self.assertEqual(scheduled_result.visible_frame, 8)
        self.assertEqual(schedule.mask[4], (False, False, False, False, False))
        self.assertEqual(schedule.lane.visible_tokens(5), scheduled_pending.fragments[0])
        self.assertEqual(schedule.lane.visible_tokens(6), scheduled_result.fragments[0])
        self.assertEqual(schedule.lane.visible_tokens(7), scheduled_result.fragments[1])
        # Building an offline schedule cannot make OBS_END visible early: the
        # same REF remains blocked through frame 7 and frees in frame 8.
        self.assertTrue(guard.is_active(0))
        with self.assertRaisesRegex(RefLeaseError, "already active"):
            ActionCompiler(self.manifest).compile_call(
                6,
                0,
                "timer.create",
                {"minutes": "MIN_4"},
                active_refs=guard,
            )
        self.assertEqual(schedule.release_visible_refs(guard, current_frame=7), ())
        self.assertTrue(guard.is_active(0))
        released = schedule.release_visible_refs(guard, current_frame=8)
        self.assertEqual(released, (lease,))
        self.assertFalse(guard.is_active(0))
        self.assertNotIn(lease.call_uuid, tuple(token for fragment in scheduled_result.symbolic_fragments for token in fragment))

    def test_fifo_uses_event_sequence_for_same_available_frame(self) -> None:
        later_sequence = self._event(
            event_id="later-seq",
            event_seq=20,
            available_frame=7,
            issued_frame=6,
            ref=1,
            kind=EnvironmentKind.PENDING,
            call_uuid="hidden-call-1",
        )
        earlier_sequence = self._event(
            event_id="earlier-seq",
            event_seq=10,
            available_frame=7,
            issued_frame=6,
            ref=0,
            kind=EnvironmentKind.PENDING,
        )

        schedule = self.scheduler.schedule(10, [later_sequence, earlier_sequence])

        self.assertEqual(
            [scheduled.event.event_id for scheduled in schedule.scheduled_events],
            ["earlier-seq", "later-seq"],
        )
        self.assertEqual(schedule.event("earlier-seq").injected_start.frame, 7)
        self.assertEqual(schedule.event("later-seq").injected_start.frame, 8)

    def test_long_observation_fragments_have_one_final_obs_end(self) -> None:
        event = self._event(
            event_id="weather",
            event_seq=3,
            available_frame=4,
            issued_frame=3,
            ref=2,
            kind=EnvironmentKind.OK,
            payload_tokens=("FORECAST_03", "TEMP_C_21"),
            tool_name="weather.lookup",
            call_uuid="hidden-weather",
        )
        fragments = event.symbolic_fragments()

        self.assertEqual(
            fragments,
            (
                ("OBS_BEGIN", "OK", "REF_2", "TOOL_WEATHER", "FORECAST_03"),
                ("OBS_MORE", "REF_2", "TEMP_C_21", "OBS_END"),
            ),
        )
        self.assertEqual(validate_environment_fragments(event, fragments, manifest=self.manifest), fragments)
        with self.assertRaisesRegex(EnvironmentFragmentError, "OBS_BEGIN"):
            validate_environment_fragments(
                event,
                (("OBS_MORE", "REF_2", "TEMP_C_21", "OBS_END"),),
                manifest=self.manifest,
            )
        with self.assertRaisesRegex(EnvironmentFragmentError, "OBS_END"):
            validate_environment_fragments(
                event,
                (
                    ("OBS_BEGIN", "OK", "REF_2", "TOOL_WEATHER", "OBS_END"),
                    ("OBS_MORE", "REF_2", "TEMP_C_21", "OBS_END"),
                ),
                manifest=self.manifest,
            )

        schedule = self.scheduler.schedule(8, [event])
        scheduled = schedule.event("weather")
        self.assertEqual((scheduled.injected_start.frame, scheduled.injected_end.frame), (4, 5))
        self.assertEqual(scheduled.visible_frame, 6)

    def test_lease_mismatch_is_rejected_without_releasing_the_ref(self) -> None:
        guard, parsed, lease = self._reserve_timer_ref()
        stale = self._event(
            event_id="stale",
            event_seq=0,
            available_frame=parsed.issued_frame + 1,
            issued_frame=parsed.issued_frame,
            ref=0,
            kind=EnvironmentKind.ERROR,
            payload_tokens=("ERR_EXECUTOR",),
            call_uuid=lease.call_uuid,
            lease=lease.lease + 1,
        )
        with self.assertRaisesRegex(RefLeaseError, "stale"):
            self.scheduler.schedule(8, [stale], lease_guard=guard)
        self.assertTrue(guard.is_active(0))

    def test_guard_binds_event_timestamps_to_the_authenticated_call_end(self) -> None:
        guard, parsed, lease = self._reserve_timer_ref()
        forged_pre_call = self._event(
            event_id="forged-pre-call",
            event_seq=0,
            available_frame=1,
            issued_frame=0,
            ref=0,
            kind=EnvironmentKind.PENDING,
            call_uuid=lease.call_uuid,
            lease=lease.lease,
        )
        with self.assertRaisesRegex(RefLeaseError, "issued_frame"):
            self.scheduler.schedule(8, [forged_pre_call], lease_guard=guard)
        self.assertTrue(guard.is_active(0))
        self.assertEqual(parsed.issued_frame, 4)

    def test_schedule_rejects_duplicates_timeline_overflow_and_scored_pad(self) -> None:
        first = self._event(
            event_id="dup-a",
            event_seq=1,
            available_frame=4,
            issued_frame=3,
            ref=0,
            kind=EnvironmentKind.PENDING,
        )
        duplicate_seq = self._event(
            event_id="dup-b",
            event_seq=1,
            available_frame=5,
            issued_frame=4,
            ref=1,
            kind=EnvironmentKind.PENDING,
            call_uuid="hidden-call-1",
        )
        with self.assertRaisesRegex(EnvironmentProtocolError, "event_seq"):
            self.scheduler.schedule(8, [first, duplicate_seq])
        with self.assertRaisesRegex(EnvironmentProtocolError, "exceeds"):
            self.scheduler.schedule(4, [first])
        with self.assertRaisesRegex(EnvironmentProtocolError, "PAD"):
            EnvironmentMicroLane(
                ids=((-1, -1, -1, -1, -1),),
                mask=((True, False, False, False, False),),
            )


if __name__ == "__main__":
    unittest.main()
