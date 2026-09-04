import copy
import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(TEST_ROOT))

import gicc_comm_group_plan_bridge as groups
import gicc_comm_structural_heuristic as heuristic
import gicc_llm_bridge as bridge
from test_gicc_comm_group_plan_bridge import PLATFORM, feature, template


SITE_IDS = [
    "unit.cpp:10:group_kernel::0",
    "unit.cpp:11:group_kernel::1",
]


def calibrated_platform():
    platform = copy.deepcopy(PLATFORM)
    platform["deployment_constraints"] = {"proxy_worker_lanes": 8}
    platform["measurements"] = {
        "proxy_fixed_us": 5.0,
        "proxy_per_op_issue_us": 3.0,
        "trigger_fixed_us": 8.0,
        "trigger_per_op_stage_us": 1.0,
    }
    return platform


def numeric_group_graph(grid_blocks, *, platform=None):
    rows = [feature(site_id) for site_id in SITE_IDS]
    for row in rows:
        row["grid_blocks"] = grid_blocks
        row["launch_grid"] = {"x": grid_blocks, "y": 1, "z": 1}
        row["launch_contexts"][0]["grid_blocks"] = grid_blocks
        row["launch_contexts"][0]["launch_grid"] = {
            "x": grid_blocks, "y": 1, "z": 1,
        }
    dossier = bridge.make_dossier(
        rows, platform if platform is not None else calibrated_platform()
    )
    return groups.make_group_graph(dossier, [template()])


def selected_candidate(graph, decision):
    opportunity = graph["opportunities"][0]
    selected_id = decision["selections"][opportunity["opportunity_id"]][
        "candidate_id"
    ]
    return next(
        candidate for candidate in opportunity["candidates"]
        if candidate["candidate_id"] == selected_id
    )


class CommunicationStructuralHeuristicTests(unittest.TestCase):
    def test_parallel_group_selects_uniform_proxy(self):
        graph = numeric_group_graph(8)
        decision, diagnostics = heuristic.make_decision(graph)
        self.assertEqual(
            "group_uniform_proxy",
            selected_candidate(graph, decision)["kind"],
        )
        diagnostic = next(iter(diagnostics.values()))
        self.assertEqual("proxy", diagnostic["selected_action"])
        self.assertEqual(2, diagnostic["proxy_concurrency"])
        self.assertLess(
            diagnostic["estimated_issue_us"]["proxy"],
            diagnostic["estimated_issue_us"]["trigger"],
        )

    def test_serial_group_selects_uniform_trigger(self):
        graph = numeric_group_graph(1)
        decision, diagnostics = heuristic.make_decision(graph)
        self.assertEqual(
            "group_uniform_trigger",
            selected_candidate(graph, decision)["kind"],
        )
        self.assertEqual(
            "trigger", next(iter(diagnostics.values()))["selected_action"]
        )

    def test_missing_launch_shape_falls_back_to_semantic_default(self):
        dossier = bridge.make_dossier(
            [feature(site_id) for site_id in SITE_IDS], calibrated_platform()
        )
        graph = groups.make_group_graph(dossier, [template()])
        decision, diagnostics = heuristic.make_decision(graph)
        self.assertEqual(
            "group_uniform_default",
            selected_candidate(graph, decision)["kind"],
        )
        self.assertEqual(
            "incomplete_or_nonuniform_compiler_shape",
            next(iter(diagnostics.values()))["reason"],
        )

    def test_incomplete_platform_calibration_falls_back(self):
        platform = calibrated_platform()
        del platform["measurements"]["proxy_per_op_issue_us"]
        graph = numeric_group_graph(8, platform=platform)
        decision, diagnostics = heuristic.make_decision(graph)
        self.assertEqual(
            "group_uniform_default",
            selected_candidate(graph, decision)["kind"],
        )
        self.assertEqual(
            "incomplete_independent_platform_calibration",
            next(iter(diagnostics.values()))["reason"],
        )

    def test_estimated_cost_tie_falls_back(self):
        platform = calibrated_platform()
        platform["measurements"]["trigger_fixed_us"] = 6.0
        graph = numeric_group_graph(8, platform=platform)
        decision, diagnostics = heuristic.make_decision(graph)
        self.assertEqual(
            "group_uniform_default",
            selected_candidate(graph, decision)["kind"],
        )
        self.assertEqual(
            "estimated_issue_cost_tie",
            next(iter(diagnostics.values()))["reason"],
        )

    def test_missing_uniform_physical_candidate_falls_back(self):
        graph = copy.deepcopy(numeric_group_graph(8))
        opportunity = graph["opportunities"][0]
        opportunity["candidates"] = [
            candidate for candidate in opportunity["candidates"]
            if candidate["kind"] != "group_uniform_trigger"
        ]
        payload = dict(graph)
        payload.pop("graph_id")
        graph["graph_id"] = bridge._fingerprint(payload)
        groups.verified_graph(graph)
        decision, diagnostics = heuristic.make_decision(graph)
        self.assertEqual(
            "group_uniform_default",
            selected_candidate(graph, decision)["kind"],
        )
        self.assertEqual(
            "uniform_physical_candidate_missing",
            next(iter(diagnostics.values()))["reason"],
        )

    def test_rule_never_selects_compiler_schedule_transform(self):
        graph = numeric_group_graph(8)
        kinds = {
            candidate["kind"]
            for candidate in graph["opportunities"][0]["candidates"]
        }
        self.assertIn("group_trigger_early", kinds)
        decision, _ = heuristic.make_decision(graph)
        candidate = selected_candidate(graph, decision)
        self.assertEqual("group_uniform_proxy", candidate["kind"])
        self.assertTrue(all(
            request["transform"] == "NONE"
            for request in candidate["materializer"]["sites"].values()
        ))

    def test_unread_evaluation_field_cannot_change_selected_candidate(self):
        original = numeric_group_graph(8)
        platform = calibrated_platform()
        platform["evaluation_runtime_us"] = {
            "invented_winner": "trigger",
        }
        changed = numeric_group_graph(8, platform=platform)
        original_decision, _ = heuristic.make_decision(original)
        changed_decision, _ = heuristic.make_decision(changed)
        self.assertEqual(
            selected_candidate(original, original_decision)["kind"],
            selected_candidate(changed, changed_decision)["kind"],
        )

    def test_control_is_deterministic_graph_bound_and_bridge_accepted(self):
        graph = numeric_group_graph(8)
        decision, hint, control = heuristic.make_control(graph)
        self.assertEqual(
            (decision, hint, control), heuristic.make_control(graph)
        )
        accepted_hint, accepted, errors = groups.plan_to_hint(graph, decision)
        self.assertTrue(accepted, errors)
        self.assertEqual(hint, accepted_hint)
        self.assertFalse(control["boundary"]["source_visible"])
        self.assertFalse(
            control["boundary"]["evaluation_runtime_results_visible"]
        )
        self.assertFalse(
            control["boundary"]["schedule_transforms_selectable_by_rule"]
        )

    def test_missing_default_on_singleton_falls_back_to_trigger(self):
        row = feature(SITE_IDS[0])
        row["legal_paths"] = ["proxy", "trigger"]
        dossier = bridge.make_dossier([row], calibrated_platform())
        graph = groups.make_group_graph(dossier, [template()])
        decision, diagnostics = heuristic.make_decision(graph)
        self.assertEqual("site_trigger", selected_candidate(graph, decision)["kind"])
        self.assertEqual(
            "incomplete_or_nonuniform_compiler_shape",
            next(iter(diagnostics.values()))["reason"],
        )


if __name__ == "__main__":
    unittest.main()
