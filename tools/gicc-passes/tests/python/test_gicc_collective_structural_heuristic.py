import copy
import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(TEST_ROOT))

import gicc_collective_plan_bridge as plans
import gicc_collective_structural_heuristic as heuristic
from test_gicc_collective_plan_bridge import ANCHOR, PROFILE, inventory


def hierarchy_descriptor(chunks):
    suffix = "" if chunks == 1 else f"_pipe{chunks}"
    return {
        "family": ANCHOR["family"],
        "contract": ANCHOR["contract"],
        "algorithm": "hierarchical_double_tree" + suffix,
        "communication_graph": "node_double_tree",
        "step_complexity": "O_log_nodes_plus_ppn",
        "topology": "two_level_node_hierarchy",
        "synchronization": "device_cooperative",
        "dynamic_guarded": "true",
        "pipeline_chunks": str(chunks),
    }


def hierarchy_inventory(*, omit_chunks=None):
    value = inventory()
    legality = {
        "same_family": True,
        "same_semantic_contract": True,
        "exact_function_type": True,
        "void_call_materializer": True,
    }
    for chunks in (1, 4, 8):
        if chunks == omit_chunks:
            continue
        descriptor = hierarchy_descriptor(chunks)
        value["opportunities"][0]["candidates"].append({
            "catalog_id": plans._target_id("catalog", descriptor),
            "descriptor": descriptor,
            "compiler_legality": legality,
        })
    return value


class CollectiveStructuralHeuristicTests(unittest.TestCase):
    def setUp(self):
        self.graph = plans.make_graph(hierarchy_inventory(), PROFILE)

    def selected_options(self, decision, graph=None):
        graph = graph or self.graph
        opportunity = graph["opportunities"][0]
        selected = decision["selections"][opportunity["opportunity_id"]][
            "slot_candidate_ids"
        ]
        return [
            next(
                option for option in slot["options"]
                if option["option_id"] == selected[slot["slot_id"]]
            )
            for slot in opportunity["decision_slots"]
        ]

    def test_n8_rule_selects_hierarchy_and_preregistered_pipeline_depth(self):
        decision, hint, control = heuristic.make_control(self.graph)
        options = self.selected_options(decision)
        self.assertEqual(
            [1, 1, 4, 8],
            [int(option["compiler_descriptor"]["pipeline_chunks"])
             for option in options],
        )
        self.assertTrue(all(heuristic._hierarchy_tree(option) for option in options))
        accepted_hint, accepted, errors = plans.decision_to_hint(
            self.graph, decision
        )
        self.assertTrue(accepted, errors)
        self.assertEqual(hint, accepted_hint)
        self.assertFalse(control["boundary"]["source_visible"])
        self.assertFalse(control["boundary"]["runtime_results_visible"])
        self.assertEqual(
            heuristic.PIPELINE_CHUNKS_BY_SLOT,
            tuple(control["rule"]["pipeline_chunks_by_ordered_slot"]),
        )

    def test_small_topology_falls_back_to_semantic_anchor(self):
        profile = copy.deepcopy(PROFILE)
        profile["topology"]["nodes"] = 2
        graph = plans.make_graph(hierarchy_inventory(), profile)
        decision = heuristic.make_decision(graph)
        self.assertTrue(all(
            option["role"] == "anchor"
            for option in self.selected_options(decision, graph)
        ))

    def test_incomplete_structural_cohort_falls_back_atomically(self):
        graph = plans.make_graph(hierarchy_inventory(omit_chunks=4), PROFILE)
        decision = heuristic.make_decision(graph)
        self.assertTrue(all(
            option["role"] == "anchor"
            for option in self.selected_options(decision, graph)
        ))

    def test_unread_evaluation_field_cannot_change_selected_option_ids(self):
        profile = copy.deepcopy(PROFILE)
        profile["evaluation_runtime_us"] = {
            "invented_label": 0.001,
        }
        graph = plans.make_graph(hierarchy_inventory(), profile)
        original = [
            option["option_id"]
            for option in self.selected_options(heuristic.make_decision(self.graph))
        ]
        changed = [
            option["option_id"]
            for option in self.selected_options(
                heuristic.make_decision(graph), graph
            )
        ]
        self.assertEqual(original, changed)

    def test_control_is_deterministic(self):
        self.assertEqual(
            heuristic.make_control(self.graph),
            heuristic.make_control(self.graph),
        )


if __name__ == "__main__":
    unittest.main()
