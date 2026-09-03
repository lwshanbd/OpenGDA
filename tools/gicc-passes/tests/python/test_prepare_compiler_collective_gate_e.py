import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PASS_ROOT / "experiments" / "collective"))

import prepare_compiler_collective_gate_e as gate_e
from test_gicc_collective_plan_bridge import PROFILE, inventory


class CompilerCollectiveGateETests(unittest.TestCase):
    def setUp(self):
        self.graph = gate_e.plans.make_graph(inventory(), PROFILE)

    def test_exact_schema_limits_output_to_existing_option_ids(self):
        schema = gate_e.exact_response_schema(self.graph)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            {"const": gate_e.plans.DECISION_SCHEMA},
            schema["properties"]["schema_version"],
        )
        opportunity = self.graph["opportunities"][0]
        selection = schema["properties"]["selections"]["properties"][
            opportunity["opportunity_id"]
        ]
        slots = selection["properties"]["slot_candidate_ids"]
        self.assertFalse(slots["additionalProperties"])
        self.assertEqual(
            [slot["slot_id"] for slot in opportunity["decision_slots"]],
            slots["required"],
        )
        for slot in opportunity["decision_slots"]:
            self.assertEqual(
                [option["option_id"] for option in slot["options"]],
                slots["properties"][slot["slot_id"]]["enum"],
            )

    def test_gate_e_requires_confirmed_paired_headroom(self):
        manifest = {"manifest_id": "sha256:" + "1" * 64}
        value = {
            "schema_version": "gicc-collective-confirmatory-analysis-v1",
            "graph_id": self.graph["graph_id"],
            "manifest_id": manifest["manifest_id"],
            "primary_headroom_confirmation": {
                "positive_point_estimate": True,
                "confidence_interval_excludes_one": True,
                "point_estimate": 1.1,
            },
            "replicate_monitors": [
                {"replicate": index} for index in (1, 2, 3)
            ],
        }
        gate_e.validate_gate_d_result(value, self.graph, manifest)
        failed = copy.deepcopy(value)
        failed["primary_headroom_confirmation"][
            "confidence_interval_excludes_one"
        ] = False
        with self.assertRaisesRegex(gate_e.GateEError, "did not confirm"):
            gate_e.validate_gate_d_result(failed, self.graph, manifest)

    def test_protocol_preregisters_rich_input_ablation_and_gbt_boundary(self):
        self.assertEqual(20, gate_e.TRIALS_PER_VIEW)
        self.assertEqual(
            ("relational", "descriptors", "opaque"),
            gate_e.plans.MODEL_VIEW_KINDS,
        )
        self.assertIn("source-free compiler facts", gate_e.SYSTEM_PROMPT)
        self.assertIn("Do not use tools, source code", gate_e.SYSTEM_PROMPT)

    def test_request_is_unauthorized_and_freezes_sixty_sequential_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory).resolve()

            def materialize(relative, value):
                path = bundle / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value)
                return path

            freeze_path = materialize("FROZEN_V3_MANIFEST.json", "{}\n")
            graph_path = materialize(
                "discovery/graph.json", json.dumps(self.graph)
            )
            manifest = {"manifest_id": "sha256:" + "1" * 64}
            manifest_path = materialize(
                "controls/manifest.json", json.dumps(manifest)
            )
            gate_d = {
                "primary_headroom_confirmation": {"point_estimate": 1.1}
            }
            gate_d_path = materialize(
                "gate-d/analysis.json", json.dumps(gate_d)
            )
            prompts = {
                view: materialize(f"prompts/{view}.txt", f"{view}\n")
                for view in gate_e.plans.MODEL_VIEW_KINDS
            }
            system = materialize("gate-e/system-prompt.txt", gate_e.SYSTEM_PROMPT)
            schema = materialize(
                "gate-e/response-schema.json",
                json.dumps(gate_e.exact_response_schema(self.graph)),
            )
            request = gate_e.build_request({
                "bundle": bundle,
                "freeze_path": freeze_path,
                "freeze": {"manifest_id": "sha256:" + "2" * 64},
                "graph_path": graph_path,
                "graph": self.graph,
                "manifest_path": manifest_path,
                "manifest": manifest,
                "gate_d_path": gate_d_path,
                "gate_d": gate_d,
                "prompts": prompts,
            }, schema, system)
            self.assertFalse(request["authorization"]["granted"])
            self.assertEqual(
                "awaiting_explicit_provider_authorization", request["status"]
            )
            self.assertEqual(
                60, request["provider_delivery"]["total_provider_calls"]
            )
            self.assertTrue(
                request["provider_delivery"]["calls_are_sequential"]
            )
            self.assertIn("ineligible", request["comparators"]["gbt"])
            payload = dict(request)
            request_id = payload.pop("request_id")
            self.assertEqual(gate_e.bridge._fingerprint(payload), request_id)


if __name__ == "__main__":
    unittest.main()
