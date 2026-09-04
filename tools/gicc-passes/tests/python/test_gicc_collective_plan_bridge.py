import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PASS_ROOT / "python"))

import gicc_collective_plan_bridge as collective


ANCHOR = {
    "family": "allreduce_sum_f32_inplace",
    "contract": "registered_buffers_flags_v1",
    "algorithm": "baseline_auto",
    "count_arg": "9",
    "ppn_arg": "10",
    "element_bytes": "4",
    "thresholds_bytes": "4096,262144,8388608",
}


def candidate(algorithm, graph, depth, topology):
    return {
        "family": ANCHOR["family"],
        "contract": ANCHOR["contract"],
        "algorithm": algorithm,
        "communication_graph": graph,
        "step_complexity": depth,
        "topology": topology,
        "synchronization": "device_cooperative",
        "dynamic_guarded": "true",
    }


CANDIDATES = [
    candidate("hierarchical_ring", "ring", "O(nodes)", "node_hierarchy"),
    candidate("flat_double_tree", "double_tree", "O(log(ranks))", "flat"),
    candidate("hierarchical_direct", "reduce_scatter", "O(ppn)", "node_hierarchy"),
]


def inventory():
    legality = {
        "same_family": True,
        "same_semantic_contract": True,
        "exact_function_type": True,
        "void_call_materializer": True,
    }
    return {
        "schema_version": collective.INVENTORY_SCHEMA,
        "compiler_only": True,
        "source_visible": False,
        "target_triple": "x86_64-unknown-linux-gnu",
        "opportunities": [{
            "opportunity_id": "opportunity:" + "1" * 24,
            "family": ANCHOR["family"],
            "contract": ANCHOR["contract"],
            "anchor_id": collective._target_id("anchor", ANCHOR),
            "anchor_descriptor": ANCHOR,
            "call_facts": {
                "count": {"kind": "dynamic_ir_expression"},
                "element_bytes": 4,
                "message_bytes": None,
                "ranks_per_node": {"kind": "constant", "value": 8},
                "loop_depth": 1,
                "arithmetic_before": 31,
                "arithmetic_after": 17,
                "direct_call": True,
                "void_return": True,
            },
            "compiler_policy_thresholds_bytes": [4096, 262144, 8388608],
            "candidates": [{
                "catalog_id": collective._target_id("candidate", descriptor),
                "descriptor": descriptor,
                "compiler_legality": legality,
            } for descriptor in CANDIDATES],
        }],
    }


PROFILE = {
    "schema_version": collective.PROFILE_SCHEMA,
    "platform_id": "unit-mi250x-cxi",
    "topology": {"nodes": 8, "ranks_per_node": 8, "gpus_per_node": 8},
    "hardware": {"gpu": "MI250X-GCD", "nics_per_node": 4},
    "transport": {"inter_node": "Slingshot-11", "intra_node": "xGMI"},
    "resource_constraints": {"cooperative_launch": True},
    "message_distribution": {"bins_weight": [0.4, 0.3, 0.2, 0.1]},
}


class CollectivePlanBridgeTests(unittest.TestCase):
    def setUp(self):
        self.graph = collective.make_graph(inventory(), PROFILE)
        self.opportunity = self.graph["opportunities"][0]

    def decision(self):
        slots = self.opportunity["decision_slots"]
        algorithms = [
            "flat_double_tree", "flat_double_tree",
            "hierarchical_direct", "hierarchical_ring",
        ]
        chosen = {}
        for slot, algorithm in zip(slots, algorithms):
            option = next(item for item in slot["options"]
                          if item["algorithm"] == algorithm)
            chosen[slot["slot_id"]] = option["option_id"]
        return {
            "schema_version": collective.DECISION_SCHEMA,
            "graph_id": self.graph["graph_id"],
            "selections": {
                self.opportunity["opportunity_id"]: {
                    "slot_candidate_ids": chosen,
                    "confidence": 0.82,
                    "rationale": "tree depth for small messages; hierarchy for bandwidth",
                }
            },
        }

    def test_graph_is_relational_source_free_and_has_joint_capacity(self):
        collective.verified_graph(self.graph)
        self.assertEqual(4, len(self.opportunity["decision_slots"]))
        self.assertEqual(4 ** 4, self.opportunity["joint_action_space_size"])
        model_view = collective._model_view(self.graph)
        model_opportunity = model_view["opportunities"][0]
        self.assertEqual(4, len(model_opportunity["candidate_classes"]))
        self.assertTrue(all(
            set(option) == {"option_id", "candidate_class_id"}
            for slot in model_opportunity["decision_slots"]
            for option in slot["allowed_options"]
        ))
        self.assertTrue(any(
            relation.get("property") == "topology"
            for relation in model_opportunity["relations"]
        ))
        descriptors = collective._model_view(self.graph, "descriptors")
        descriptor_opportunity = descriptors["opportunities"][0]
        self.assertEqual([], descriptor_opportunity["relations"])
        self.assertEqual(4, len(descriptor_opportunity["candidate_classes"]))
        opaque = collective._model_view(self.graph, "opaque")
        opaque_opportunity = opaque["opportunities"][0]
        self.assertEqual([], opaque_opportunity["candidate_classes"])
        self.assertTrue(all(
            set(option) == {"option_id"}
            for slot in opaque_opportunity["decision_slots"]
            for option in slot["allowed_options"]
        ))
        prompt = collective.render_prompt(self.graph)
        self.assertNotIn("target_id", prompt)
        self.assertNotIn("SECRET_SOURCE_SENTINEL", prompt)
        self.assertIn("step_complexity", prompt)
        self.assertEqual(1, prompt.count('"algorithm": "flat_double_tree"'))
        self.assertEqual(3, len({
            collective.render_prompt(self.graph, kind)
            for kind in collective.MODEL_VIEW_KINDS
        }))
        self.assertEqual(
            "compiler-generated option IDs only",
            self.graph["boundary"]["model_output"],
        )

    def test_option_ids_expand_only_to_a_compiler_hint(self):
        hint, accepted, errors = collective.decision_to_hint(
            self.graph, self.decision()
        )
        self.assertTrue(accepted, errors)
        selection = hint["selections"][self.opportunity["opportunity_id"]]
        self.assertEqual("size_policy", selection["kind"])
        self.assertEqual([4096, 262144, 8388608, None],
                         [rule["max_bytes"] for rule in selection["rules"]])
        self.assertTrue(hint["llm_metadata"]["compiler_only_output"])
        self.assertTrue(selection["candidate_id"].startswith("candidate:"))

    def test_response_schema_allows_only_graph_bound_options(self):
        schema = collective.decision_response_schema(self.graph)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            {"const": self.graph["graph_id"]},
            schema["properties"]["graph_id"],
        )
        selections = schema["properties"]["selections"]
        self.assertEqual(
            [self.opportunity["opportunity_id"]], selections["required"]
        )
        selection = selections["properties"][
            self.opportunity["opportunity_id"]
        ]
        slots = selection["properties"]["slot_candidate_ids"]
        for slot in self.opportunity["decision_slots"]:
            self.assertEqual(
                [option["option_id"] for option in slot["options"]],
                slots["properties"][slot["slot_id"]]["enum"],
            )
        rendered = json.dumps(schema, sort_keys=True)
        self.assertNotIn("target_id", rendered)

    def test_invented_option_and_code_fail_closed_to_anchor(self):
        decision = self.decision()
        entry = decision["selections"][self.opportunity["opportunity_id"]]
        first = next(iter(entry["slot_candidate_ids"]))
        entry["slot_candidate_ids"][first] = "option:" + "0" * 24
        entry["llvm_ir"] = "call void @invented()"
        hint, accepted, errors = collective.decision_to_hint(self.graph, decision)
        self.assertFalse(accepted)
        self.assertTrue(any("unknown field" in error for error in errors))
        self.assertTrue(any("not compiler-generated" in error for error in errors))
        selection = hint["selections"][self.opportunity["opportunity_id"]]
        self.assertEqual(
            inventory()["opportunities"][0]["anchor_id"],
            selection["target_id"],
        )
        self.assertEqual("uniform", selection["kind"])

    def test_tampered_threshold_or_descriptor_breaks_graph_hash(self):
        graph = copy.deepcopy(self.graph)
        graph["opportunities"][0]["decision_slots"][0]["message_bytes"][
            "max"
        ] = 8192
        with self.assertRaisesRegex(collective.CollectivePlanError, "graph_id"):
            collective.verified_graph(graph)

    def test_inventory_rejects_tampered_catalog_id(self):
        value = inventory()
        value["opportunities"][0]["candidates"][0]["catalog_id"] = (
            "catalog:" + "0" * 24
        )
        with self.assertRaisesRegex(collective.CollectivePlanError, "catalog_id"):
            collective.make_graph(value, PROFILE)

    def test_platform_can_mask_an_unstable_algorithm(self):
        profile = copy.deepcopy(PROFILE)
        profile["disabled_algorithms"] = ["flat_double_tree"]
        graph = collective.make_graph(inventory(), profile)
        opportunity = graph["opportunities"][0]
        self.assertEqual(3 ** 4, opportunity["joint_action_space_size"])
        self.assertNotIn(
            "flat_double_tree",
            {option["algorithm"]
             for option in opportunity["decision_slots"][0]["options"]},
        )

    def test_primitive_calibration_must_be_label_disjoint_and_hash_pinned(self):
        profile = copy.deepcopy(PROFILE)
        profile["primitive_calibration"] = {
            "schema_version": "gicc-collective-primitive-calibration-v1",
            "role": "prior primitive evidence",
            "collective_action_labels_visible": False,
            "applicability": "cross-node link prior",
            "artifacts": [{"sha256": "a" * 64, "role": "calibration"}],
            "measurements": {"link_bandwidth_GB_s": 24.0},
        }
        graph = collective.make_graph(inventory(), profile)
        shown = graph["opportunities"][0]["compiler_facts"]
        self.assertEqual(
            24.0,
            shown["primitive_calibration"]["measurements"][
                "link_bandwidth_GB_s"
            ],
        )
        profile["primitive_calibration"][
            "collective_action_labels_visible"
        ] = True
        with self.assertRaisesRegex(
                collective.CollectivePlanError, "label-disjoint"):
            collective.make_graph(inventory(), profile)

    def test_calibration_artifact_content_is_verified_before_emit(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "calibration.json"
            artifact.write_text('{"prior":"primitive only"}\n')
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            profile = copy.deepcopy(PROFILE)
            profile["primitive_calibration"] = {
                "schema_version": "gicc-collective-primitive-calibration-v1",
                "role": "prior primitive evidence",
                "collective_action_labels_visible": False,
                "applicability": "cross-node link prior",
                "artifacts": [{"sha256": digest, "role": "calibration"}],
                "measurements": {"link_bandwidth_GB_s": 24.0},
            }
            verified = collective._verified_profile(profile)
            collective._verify_calibration_artifacts(verified, [artifact])
            with self.assertRaisesRegex(
                    collective.CollectivePlanError, "do not match"):
                collective._verify_calibration_artifacts(verified, [])

    def test_ids_and_graph_are_deterministic(self):
        self.assertEqual(self.graph, collective.make_graph(inventory(), PROFILE))


if __name__ == "__main__":
    unittest.main()
