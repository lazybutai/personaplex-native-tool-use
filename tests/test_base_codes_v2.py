from __future__ import annotations

import unittest

from personaplex_agent.base_codes_v2 import (
    BaseCodesV2Error,
    PersonaPlexBaseCodes,
    assemble_personaplex_base_codes,
)


class PersonaPlexBaseCodesTests(unittest.TestCase):
    def test_real_codebooks_are_placed_in_the_fixed_seventeen_stream_layout(self) -> None:
        silence = tuple(range(100, 108))
        user = tuple(tuple(10 * codebook + frame for frame in range(3)) for codebook in range(8))
        assistant = tuple(
            tuple(200 + 10 * codebook + frame for frame in range(2))
            for codebook in range(8)
        )
        record = assemble_personaplex_base_codes(
            8,
            silence_tokens=silence,
            user_audio_codes=user,
            user_start_frame=1,
            assistant_audio_codes=assistant,
            assistant_start_frame=5,
        )

        self.assertEqual(record.shape, (17, 8))
        self.assertEqual(record.codes[0], (3,) * 8)
        self.assertEqual(record.codes[1][5:7], assistant[0])
        self.assertEqual(record.codes[9][1:4], user[0])
        self.assertEqual(record.codes[1][0], silence[0])
        self.assertEqual(record.codes[9][7], silence[0])
        self.assertEqual(record.user_audio_span, (1, 4))
        self.assertEqual(record.assistant_audio_span, (5, 7))
        self.assertEqual(record.encoding_mode, "causal_streaming_80ms")

    def test_same_codes_have_a_stable_fingerprint(self) -> None:
        arguments = dict(
            frame_count=4,
            silence_tokens=tuple(range(8)),
            user_audio_codes=tuple((index, index + 1) for index in range(8)),
        )
        self.assertEqual(
            assemble_personaplex_base_codes(**arguments).fingerprint,
            assemble_personaplex_base_codes(**arguments).fingerprint,
        )

    def test_invalid_shapes_ranges_and_timeline_overflow_are_rejected(self) -> None:
        silence = tuple(range(8))
        valid_user = tuple((index, index + 1) for index in range(8))
        with self.assertRaises(BaseCodesV2Error):
            assemble_personaplex_base_codes(
                4,
                silence_tokens=silence,
                user_audio_codes=valid_user[:7],
            )
        with self.assertRaises(BaseCodesV2Error):
            assemble_personaplex_base_codes(
                2,
                silence_tokens=silence,
                user_audio_codes=valid_user,
                user_start_frame=1,
            )
        with self.assertRaises(BaseCodesV2Error):
            assemble_personaplex_base_codes(
                4,
                silence_tokens=silence,
                user_audio_codes=((2048,),) * 8,
            )
        with self.assertRaises(BaseCodesV2Error):
            PersonaPlexBaseCodes(
                codes=((0,) * 2,) * 16,
                user_audio_span=(0, 1),
                assistant_audio_span=None,
            )


if __name__ == "__main__":
    unittest.main()
