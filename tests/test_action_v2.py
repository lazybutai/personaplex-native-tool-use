from __future__ import annotations

import unittest

from personaplex_agent.action_v2 import (
    ACTION_V2_VERSION,
    ActionCompiler,
    ActionGrammarError,
    ActionKind,
    ActionPacketV2,
    ActionParser,
    ActionV2Manifest,
    ControlAction,
    DEFAULT_MANIFEST,
    MICRO_TOKENS_PER_FRAME,
    MicroLane,
    MicroPosition,
    PAD_TOKEN_ID,
    RefLeaseError,
    RefLeaseGuard,
    TerminalLeaseReceipt,
    materialize_action_lane,
)


class ActionV2ManifestTests(unittest.TestCase):
    def test_fixed_maps_are_complete_immutable_and_fingerprinted(self) -> None:
        manifest = DEFAULT_MANIFEST

        self.assertEqual(manifest.version, ACTION_V2_VERSION)
        self.assertEqual(manifest.action_cardinality, 256)
        self.assertEqual(manifest.environment_cardinality, 256)
        self.assertEqual(sorted(manifest.action_tokens.values()), list(range(256)))
        self.assertEqual(sorted(manifest.environment_tokens.values()), list(range(256)))
        self.assertEqual(manifest.action_id("NOOP"), 0)
        self.assertEqual(manifest.environment_id("OBS_BEGIN"), 0)
        self.assertEqual(manifest.action_name(15), "TOOL_HOME_SET_TEMPERATURE")
        self.assertEqual(manifest.environment_name(19), "TOOL_HOME")

        with self.assertRaises(TypeError):
            manifest.action_tokens["NOOP"] = 99  # type: ignore[index]

        snapshot = manifest.to_dict()
        snapshot["action_tokens"]["NOOP"] = 99  # type: ignore[index]
        self.assertEqual(manifest.action_id("NOOP"), 0)
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            ActionV2Manifest.from_dict(snapshot)

        cloned = ActionV2Manifest.from_dict(DEFAULT_MANIFEST.to_dict())
        self.assertEqual(cloned.fingerprint, manifest.fingerprint)
        self.assertEqual(cloned.tool_catalog_hash, manifest.tool_catalog_hash)
        with self.assertRaisesRegex(ValueError, "requires version"):
            ActionV2Manifest(version="ppx-action-v2.test")
        with self.assertRaisesRegex(ValueError, "tool catalog"):
            ActionV2Manifest(tool_schemas=(manifest.tool("timer.create"),))


class ActionV2GrammarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = DEFAULT_MANIFEST
        self.compiler = ActionCompiler(self.manifest)
        self.parser = ActionParser(self.manifest)

    def test_compiler_parser_round_trip_issues_at_call_end_micro_slot(self) -> None:
        packet = self.compiler.compile_call(
            4,
            "REF_2",
            "home.set_temperature",
            {"room": "ROOM_3", "temperature_c": "TEMP_C_22"},
        )

        self.assertEqual(
            packet.token_ids,
            (
                self.manifest.action_id("CALL_BEGIN"),
                self.manifest.action_id("TOOL_HOME_SET_TEMPERATURE"),
                self.manifest.action_id("REF_2"),
                self.manifest.action_id("ROOM_3"),
                self.manifest.action_id("TEMP_C_22"),
                self.manifest.action_id("CALL_END"),
            ),
        )
        parsed = self.parser.parse(packet)

        self.assertEqual(parsed.kind, ActionKind.CALL)
        self.assertTrue(parsed.dispatchable)
        self.assertEqual(parsed.ref, 2)
        self.assertEqual(parsed.tool_name, "home.set_temperature")
        self.assertEqual(dict(parsed.arguments), {"room": "ROOM_3", "temperature_c": "TEMP_C_22"})
        self.assertEqual((parsed.issued_frame, parsed.issued_micro), (5, 0))

    def test_call_end_at_micro_four_still_issues_in_the_first_frame(self) -> None:
        packet = self.compiler.compile_call(
            8,
            0,
            "timer.create",
            {"minutes": "MIN_5"},
        )
        parsed = self.parser.parse(packet)

        self.assertEqual(packet.token_count, 5)
        self.assertEqual((parsed.issued_frame, parsed.issued_micro), (8, 4))

    def test_fsa_rejects_odd_start_pad_reserved_incomplete_or_busy_ref(self) -> None:
        with self.assertRaisesRegex(ValueError, "even"):
            ActionPacketV2(1, (self.manifest.action_id("NOOP"),))
        with self.assertRaisesRegex(ValueError, "PAD"):
            ActionPacketV2(0, (PAD_TOKEN_ID,))
        with self.assertRaisesRegex(ValueError, "0..255"):
            ActionPacketV2(0, (256,))

        incomplete = ActionPacketV2(
            0,
            (
                self.manifest.action_id("CALL_BEGIN"),
                self.manifest.action_id("TOOL_TIMER_CREATE"),
                self.manifest.action_id("REF_0"),
                self.manifest.action_id("MIN_4"),
            ),
        )
        with self.assertRaisesRegex(ActionGrammarError, "CALL_END"):
            self.parser.parse(incomplete)

        with self.assertRaisesRegex(ActionGrammarError, "reserved"):
            self.parser.parse(ActionPacketV2(0, (192,)))

        packet = self.compiler.compile_call(
            2, 1, "weather.lookup", {"city": "CITY_7"}
        )
        with self.assertRaisesRegex(RefLeaseError, "already active"):
            self.parser.parse(packet, active_refs={1})

    def test_fixed_schema_order_controls_and_cancel_grammar(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly"):
            self.compiler.compile_call(
                0,
                0,
                "home.set_temperature",
                {"temperature_c": "TEMP_C_20"},
            )
        with self.assertRaisesRegex(ValueError, "fixed symbolic"):
            self.compiler.compile_call(
                0,
                0,
                "timer.create",
                {"minutes": "CITY_1"},
            )

        control = self.compiler.compile_control(10, ControlAction.PAUSE)
        parsed_control = self.parser.parse(control)
        self.assertEqual(parsed_control.kind, ActionKind.CONTROL)
        self.assertEqual(parsed_control.control, "PAUSE")
        self.assertFalse(parsed_control.dispatchable)

        cancel = self.compiler.compile_cancel(12, 3, active_refs={3})
        parsed_cancel = self.parser.parse(cancel, active_refs={3})
        self.assertEqual(parsed_cancel.kind, ActionKind.CANCEL)
        self.assertEqual((parsed_cancel.issued_frame, parsed_cancel.issued_micro), (12, 2))
        with self.assertRaisesRegex(RefLeaseError, "not active"):
            self.parser.parse(cancel, active_refs=set())


class ActionV2LaneAndLeaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = DEFAULT_MANIFEST
        self.compiler = ActionCompiler(self.manifest)
        self.parser = ActionParser(self.manifest)

    def test_micro_lane_preserves_all_slots_and_keeps_pad_unmasked(self) -> None:
        packet = self.compiler.compile_call(
            4,
            2,
            "home.set_temperature",
            {"room": "ROOM_3", "temperature_c": "TEMP_C_22"},
        )
        lane = materialize_action_lane(8, [packet], manifest=self.manifest)

        self.assertEqual(lane.frame_count, 8)
        self.assertEqual(lane.ids[0], (0, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID))
        self.assertEqual(lane.mask[0], (True, False, False, False, False))
        self.assertEqual(lane.visible_tokens(4), packet.token_ids[:MICRO_TOKENS_PER_FRAME])
        self.assertEqual(lane.visible_tokens(5), packet.token_ids[MICRO_TOKENS_PER_FRAME:])
        self.assertEqual(lane.mask[4], (True,) * MICRO_TOKENS_PER_FRAME)
        self.assertEqual(lane.mask[5], (True, False, False, False, False))
        self.assertTrue(all(token != PAD_TOKEN_ID for token in lane.visible_tokens(5)))
        with self.assertRaisesRegex(ValueError, "0..255"):
            MicroLane(ids=((256, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID),), mask=((True, False, False, False, False),))

        first_frame_only = self.compiler.compile_call(
            0, 1, "timer.create", {"minutes": "MIN_2"}
        )
        short_lane = materialize_action_lane(4, [first_frame_only], manifest=self.manifest)
        self.assertEqual(short_lane.visible_tokens(1), (self.manifest.action_id("NOOP"),))
        self.assertEqual(short_lane.mask[1], (True, False, False, False, False))

    def test_lane_rejects_collision_and_missing_second_physical_frame(self) -> None:
        first = self.compiler.compile_call(2, 0, "timer.create", {"minutes": "MIN_3"})
        second = self.compiler.compile_call(2, 1, "weather.lookup", {"city": "CITY_2"})
        with self.assertRaisesRegex(ValueError, "collide"):
            materialize_action_lane(8, [first, second], manifest=self.manifest)

        edge = self.compiler.compile_noop(4)
        with self.assertRaisesRegex(ValueError, "two-frame"):
            materialize_action_lane(5, [edge], manifest=self.manifest)

    def test_lease_guard_hides_uuid_and_blocks_reuse_until_terminal_visible(self) -> None:
        packet = self.compiler.compile_call(4, 0, "timer.create", {"minutes": "MIN_3"})
        parsed = self.parser.parse(packet)
        guard = RefLeaseGuard(self.manifest)
        lease = guard.reserve_parsed(parsed, call_uuid="541c27a0-40dc-5d2d-b6f0-cf9e319b2a0b")

        self.assertEqual(guard.active_refs, frozenset({0}))
        self.assertNotIn(lease.call_uuid, parsed.to_dict())
        with self.assertRaisesRegex(RefLeaseError, "already active"):
            self.parser.parse(packet, active_refs=guard)
        with self.assertRaisesRegex(RefLeaseError, "stale"):
            guard.validate_event(0, call_uuid=lease.call_uuid, lease=lease.lease + 1)
        with self.assertRaisesRegex(RefLeaseError, "available strictly after"):
            guard.release_terminal(
                TerminalLeaseReceipt(
                    ref=0,
                    call_uuid=lease.call_uuid,
                    lease=lease.lease,
                    available_frame=parsed.issued_frame,
                    injected_end=MicroPosition(parsed.issued_frame, 4),
                )
            )

        guard.release_terminal(
            TerminalLeaseReceipt(
                ref=0,
                call_uuid=lease.call_uuid,
                lease=lease.lease,
                available_frame=parsed.issued_frame + 1,
                injected_end=MicroPosition(parsed.issued_frame + 1, 4),
            )
        )
        self.assertFalse(guard.is_active(0))
        self.assertEqual(
            guard.reserve(0, call_uuid="next-hidden-uuid", issued_frame=8).lease,
            lease.lease + 1,
        )


if __name__ == "__main__":
    unittest.main()
