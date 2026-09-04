import json
import sys
import tempfile
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PASS_ROOT / "python"))

import gicc_collective_plan_bridge as collective
import gicc_comm_group_plan_bridge as communication
import gicc_comm_plan_bridge as structural
import gicc_compiler_decision_suite as suite


def communication_graph():
    site_id = "unit.cpp:1:unit_kernel::0"
    candidates = [
        communication._candidate(
            f"site_{action}", [site_id], (action,),
            "Use one compiler-advertised route.",
            ["the compiler proved this route legal"],
            {
                "site_actions": {site_id: action},
                "trigger_placement": "original_completion",
                "host_descriptor_sites": int(action == "trigger"),
                "device_proxy_sites": int(action == "proxy"),
            },
        )
        for action in ("proxy", "trigger")
    ]
    opportunity = {
        "opportunity_id": "group-opportunity:" + "1" * 24,
        "kind": "single_site_communication_route",
        "site_ids": [site_id],
        "compiler_facts": {
            "kernel": "unit_kernel",
            "operation_order": [site_id],
            "group_size": 1,
            "batch_size": [1],
            "fan_out": [1],
            "completion": {"site_id": None, "kind": None},
            "argument_relations": {"size": "single_expression"},
            "site_argument_expressions": {
                site_id: {"size": {"kind": "param", "param": 1}}
            },
            "site_legal_actions": {site_id: ["proxy", "trigger"]},
            "site_semantic_facts": {
                site_id: {"op_kind": "put_no_db", "size_kind": "param"}
            },
            "launch_contexts": {site_id: []},
            "compute_region": {
                "site_flops_to_completion": {site_id: 4},
                "distance_exact": True,
                "flops_after_last_site": 4,
            },
            "dependence_legality": {
                "group_early_trigger_legal": False,
            },
        },
        "masked_candidates": [],
        "candidates": candidates,
    }
    payload = {
        "schema_version": communication.GRAPH_SCHEMA,
        "compiler_inputs": {
            "dossier_id": "sha256:" + "2" * 64,
            "kernel_template_ids": ["sha256:" + "3" * 64],
        },
        "boundary": {
            "source_visible": False,
            "model_may_generate_code": False,
            "model_may_generate_ir": False,
            "model_may_assert_legality": False,
            "model_output": "candidate IDs only",
            "compiler_revalidates_before_materialization": True,
        },
        "objective": {
            "metric": "end_to_end_wall_time",
            "instruction": "Select one existing compiler plan.",
        },
        "platform_profile": {
            "schema_version": "gicc-platform-profile-v1",
            "platform_id": "unit-platform",
        },
        "fixed_sites": [],
        "opportunities": [opportunity],
    }
    return {**payload, "graph_id": communication._fingerprint(payload)}


def collective_graph():
    anchor = {
        "family": "allreduce",
        "contract": "sum_f32",
        "algorithm": "baseline_auto",
    }
    candidate = {
        "family": "allreduce",
        "contract": "sum_f32",
        "algorithm": "double_tree",
    }
    inventory = {
        "schema_version": collective.INVENTORY_SCHEMA,
        "compiler_only": True,
        "source_visible": False,
        "target_triple": "amdgcn-amd-amdhsa",
        "opportunities": [{
            "opportunity_id": "opportunity:" + "4" * 24,
            "family": "allreduce",
            "contract": "sum_f32",
            "anchor_descriptor": anchor,
            "anchor_id": collective._target_id("anchor", anchor),
            "call_facts": {"element_bytes": 4, "message_shape": "dynamic"},
            "compiler_policy_thresholds_bytes": [4096],
            "candidates": [{
                "catalog_id": collective._target_id("catalog", candidate),
                "descriptor": candidate,
                "compiler_legality": {
                    "same_family": True,
                    "same_semantic_contract": True,
                    "exact_function_type": True,
                    "void_call_materializer": True,
                },
            }],
        }],
    }
    profile = {
        "schema_version": collective.PROFILE_SCHEMA,
        "platform_id": "unit-collective-platform",
        "topology": {"nodes": 2, "ranks_per_node": 2},
    }
    return collective.make_graph(inventory, profile)


def structural_graph():
    site = {"site_id": "unit.cpp:7:structural_kernel::0"}
    candidates = [
        structural._candidate(
            site, kind, request["dispatch"], request["transform"],
            f"Compiler-generated {kind} plan.",
            {"network_operations": operations, "host_descriptors": descriptors},
            ["compiler proof"],
        )
        for kind, request, operations, descriptors in (
            (
                "proxy_device",
                structural._MATERIALIZER_FOR_KIND["proxy_device"], 8, 0,
            ),
            (
                "trigger_descriptor_batch",
                structural._MATERIALIZER_FOR_KIND[
                    "trigger_descriptor_batch"
                ], 8, 8,
            ),
            (
                "trigger_coalesced_loop",
                structural._MATERIALIZER_FOR_KIND[
                    "trigger_coalesced_loop"
                ], 1, 1,
            ),
            (
                "trigger_coalesced_early",
                structural._MATERIALIZER_FOR_KIND[
                    "trigger_coalesced_early"
                ], 1, 1,
            ),
        )
    ]
    opportunity = {
        "opportunity_id": structural._opportunity_id(site["site_id"]),
        "kind": "loop_communication_plan",
        "site_ids": [site["site_id"]],
        "compiler_facts": {
            "kernel": "structural_kernel",
            "size_bytes": 1024,
            "trip_count": 8,
            "batch_size": 8,
            "grid_blocks": 2,
            "descriptor_reusable": False,
            "buffer_reusable": True,
            "coalescable": True,
            "guard_kind": "always",
            "flops_to_first_use": 32,
        },
        "candidates": candidates,
    }
    payload = {
        "schema_version": structural.GRAPH_SCHEMA,
        "compiler_dossier_id": "sha256:" + "5" * 64,
        "boundary": {
            "source_visible": False,
            "model_may_generate_code": False,
            "model_may_generate_ir": False,
            "model_output": "candidate IDs only",
            "compiler_revalidates_before_materialization": True,
        },
        "objective": {
            "metric": "end_to_end_wall_time",
            "instruction": "Select one existing compiler plan.",
        },
        "platform_profile": {
            "schema_version": "gicc-platform-profile-v1",
            "platform_id": "unit-structural-platform",
        },
        "fixed_sites": [],
        "opportunities": [opportunity],
    }
    return {**payload, "graph_id": structural._fingerprint(payload)}


class CompilerDecisionSuiteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.communication_path = self.root / "communication.json"
        self.collective_path = self.root / "collective.json"
        self.structural_path = self.root / "structural.json"
        self.communication_path.write_text(
            json.dumps(communication_graph()), encoding="utf-8"
        )
        self.collective_path.write_text(
            json.dumps(collective_graph()), encoding="utf-8"
        )
        self.structural_path.write_text(
            json.dumps(structural_graph()), encoding="utf-8"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def make(self, *, prompt_dir=None):
        return suite.make_suite(
            [("unit-communication", self.communication_path)],
            [("unit-collective", self.collective_path)],
            [("unit-structural", self.structural_path)],
            prompt_dir=prompt_dir,
        )

    def test_suite_indexes_independent_families_and_three_views(self):
        prompt_dir = self.root / "prompts"
        value = self.make(prompt_dir=prompt_dir)
        self.assertEqual(value, suite.verified_suite(value, prompt_dir))
        self.assertEqual(suite.SUITE_SCHEMA, value["schema_version"])
        self.assertTrue(value["boundary"]["compiler_lto_decisions_only"])
        self.assertFalse(value["boundary"]["source_visible"])
        self.assertFalse(value["composition"]["cross_entry_joint_selection"])
        self.assertEqual(3, value["entry_count"])
        entries = {entry["label"]: entry for entry in value["entries"]}
        self.assertEqual(
            2,
            entries["unit-communication"]["decision_space"]
                ["independent_policy_count"],
        )
        self.assertEqual(
            4,
            entries["unit-collective"]["decision_space"]
                ["independent_policy_count"],
        )
        self.assertEqual(
            4,
            entries["unit-structural"]["decision_space"]
                ["independent_policy_count"],
        )
        for entry in entries.values():
            self.assertEqual(set(suite.VIEW_KINDS), set(entry["views"]))
            for view_kind, record in entry["views"].items():
                self.assertRegex(record["prompt_sha256"], r"^[0-9a-f]{64}$")
                prompt = self.root / "prompts" / entry["label"] / f"{view_kind}.txt"
                self.assertTrue(prompt.is_file())
                self.assertEqual(record["prompt_bytes"], prompt.stat().st_size)
                self.assertNotIn("unit.cpp", prompt.read_text())
            schema_record = entry["response_schema"]
            schema = (
                prompt_dir / entry["label"] / "response-schema.json"
            )
            self.assertTrue(schema.is_file())
            self.assertEqual(schema_record["bytes"], schema.stat().st_size)
            self.assertEqual(
                schema_record["file_sha256"], suite._file_sha256(schema)
            )

        prompt = prompt_dir / "unit-communication" / "opaque.txt"
        prompt.write_text(prompt.read_text() + "tampered", encoding="utf-8")
        with self.assertRaisesRegex(suite.SuiteError, "prompt content"):
            suite.verified_suite(value, prompt_dir)

        self.make(prompt_dir=prompt_dir)
        schema = prompt_dir / "unit-collective" / "response-schema.json"
        schema.write_text(schema.read_text() + "tampered", encoding="utf-8")
        with self.assertRaisesRegex(suite.SuiteError, "response schema content"):
            suite.verified_suite(value, prompt_dir)

    def test_suite_is_deterministic(self):
        self.assertEqual(self.make(), self.make())

    def test_tampered_suite_id_is_rejected(self):
        value = json.loads(json.dumps(self.make()))
        value["composition"]["cross_entry_joint_selection"] = True
        with self.assertRaisesRegex(suite.SuiteError, "suite_id"):
            suite.verified_suite(value)

    def test_rehashed_non_compiler_boundary_is_rejected(self):
        value = json.loads(json.dumps(self.make()))
        value["boundary"]["source_visible"] = True
        payload = dict(value)
        payload.pop("suite_id")
        value["suite_id"] = suite.bridge._fingerprint(payload)
        with self.assertRaisesRegex(suite.SuiteError, "compiler-only"):
            suite.verified_suite(value)

    def test_duplicate_label_is_rejected(self):
        with self.assertRaisesRegex(suite.SuiteError, "unique"):
            suite.make_suite(
                [("duplicate", self.communication_path)],
                [("duplicate", self.collective_path)],
            )

    def test_tampered_graph_is_rejected(self):
        value = json.loads(self.communication_path.read_text())
        value["objective"]["metric"] = "tampered"
        self.communication_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(communication.GroupPlanError, "graph_id"):
            self.make()


if __name__ == "__main__":
    unittest.main()
