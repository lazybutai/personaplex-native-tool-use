from __future__ import annotations

import json
import unittest
from pathlib import Path

from personaplex_agent.action_v2 import DEFAULT_MANIFEST


class ActionV2SchemaSnapshotTests(unittest.TestCase):
    def test_schema_is_pinned_to_the_runtime_manifest(self) -> None:
        schema_path = Path(__file__).parents[1] / "schemas" / "ppx_action_v2.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        properties = schema["properties"]
        self.assertEqual(properties["protocol_version"]["const"], DEFAULT_MANIFEST.version)
        self.assertEqual(
            properties["manifest_fingerprint"]["const"],
            DEFAULT_MANIFEST.fingerprint,
        )
        self.assertEqual(
            properties["tool_catalog_hash"]["const"],
            DEFAULT_MANIFEST.tool_catalog_hash,
        )
        self.assertEqual(properties["frame_ms"]["const"], DEFAULT_MANIFEST.base_frame_ms)
        self.assertEqual(
            properties["action_slots_per_frame"]["const"],
            DEFAULT_MANIFEST.micro_tokens_per_frame,
        )
        self.assertEqual(
            properties["environment_slots_per_frame"]["const"],
            DEFAULT_MANIFEST.micro_tokens_per_frame,
        )

    def test_schema_tool_names_match_the_fixed_catalog(self) -> None:
        schema_path = Path(__file__).parents[1] / "schemas" / "ppx_action_v2.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        event_tools = set(schema["$defs"]["event"]["properties"]["tool_name"]["enum"])
        manifest_tools = {tool.name for tool in DEFAULT_MANIFEST.tool_schemas}
        self.assertEqual(event_tools, manifest_tools)


if __name__ == "__main__":
    unittest.main()
