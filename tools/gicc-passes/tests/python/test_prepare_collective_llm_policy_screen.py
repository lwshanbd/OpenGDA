import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = Path(__file__).resolve().parent
COLLECTIVE = PASS_ROOT / "experiments" / "collective"
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(PASS_ROOT / "experiments"))
sys.path.insert(0, str(COLLECTIVE))
sys.path.insert(0, str(TEST_ROOT))

import analyze_compiler_llm_capability_trials as capability_analysis
import gicc_collective_plan_bridge as plans
import gicc_compiler_policy_bridge as policy_bridge
import prepare_collective_llm_policy_screen as screen
from test_gicc_collective_plan_bridge import PROFILE, inventory


def control_rows(graph):
    algorithms = sorted({
        option["algorithm"]
        for option in graph["opportunities"][0]["decision_slots"][0]["options"]
    })
    by_bin = {
        "baseline_auto": [100.0, 100.0, 100.0, 100.0],
        "flat_double_tree": [80.0, 90.0, 110.0, 120.0],
        "hierarchical_direct": [120.0, 80.0, 90.0, 110.0],
        "hierarchical_ring": [130.0, 120.0, 80.0, 70.0],
    }
    slots = graph["opportunities"][0]["decision_slots"]

    def slot_index(size):
        for index, slot in enumerate(slots):
            lower = slot["message_bytes"]["min"]
            upper = slot["message_bytes"]["max"]
            if ((lower is None or size >= lower)
                    and (upper is None or size <= upper)):
                return index
        raise AssertionError(size)

    return {
        replicate: {
            algorithm: {
                str(size): by_bin[algorithm][slot_index(size)]
                * (1.0 + 0.01 * (replicate - 2))
                for size in screen.collective_eval.GATE_B_SIZES
            }
            for algorithm in algorithms
        }
        for replicate in (1, 2, 3)
    }


def selection(graph, algorithms):
    opportunity = graph["opportunities"][0]
    return {
        f"{opportunity['opportunity_id']}/{slot['slot_id']}": next(
            option["option_id"] for option in slot["options"]
            if option["algorithm"] == algorithm
        )
        for slot, algorithm in zip(
            opportunity["decision_slots"], algorithms, strict=True,
        )
    }


class CollectiveLlmPolicyScreenTests(unittest.TestCase):
    def setUp(self):
        self.graph = plans.make_graph(inventory(), PROFILE)
        self.rows = control_rows(self.graph)
        self.controls = screen.analyze_control_rows(self.graph, self.rows)

    def test_full_catalog_rows_define_weighted_bin_oracle(self):
        self.assertEqual(3, self.controls["replicates"])
        self.assertEqual(
            [
                "flat_double_tree",
                "hierarchical_direct",
                "hierarchical_ring",
                "hierarchical_ring",
            ],
            [item["algorithm"] for item in self.controls["oracle_bins"]],
        )
        weights = self.controls["unit_weights"]
        self.assertAlmostEqual(1.0, sum(weights.values()))
        first_bin_weight = sum(
            weight for unit, weight in weights.items()
            if unit in {"message_bytes:1024", "message_bytes:4096"}
        )
        self.assertAlmostEqual(0.4, first_bin_weight)

    def test_policy_cost_composes_only_its_graph_bound_bin_choices(self):
        mixed = selection(self.graph, [
            "flat_double_tree", "hierarchical_direct",
            "baseline_auto", "hierarchical_ring",
        ])
        costs = screen.policy_cost(self.graph, mixed, self.controls)
        self.assertEqual(80.0, costs["message_bytes:1024"])
        self.assertEqual(80.0, costs["message_bytes:65536"])
        self.assertEqual(100.0, costs["message_bytes:4194304"])
        self.assertEqual(70.0, costs["message_bytes:16777216"])

    def test_screen_covers_every_observed_policy_and_keeps_runtime_boundary(self):
        anchor = policy_bridge.decision_to_policy(
            self.graph, None
        )["selected_ids_by_slot"]
        mixed = selection(self.graph, [
            "flat_double_tree", "hierarchical_direct",
            "baseline_auto", "hierarchical_ring",
        ])
        anchor_id = policy_bridge.verified_policy(
            self.graph, anchor
        )["policy_id"]
        mixed_id = policy_bridge.verified_policy(
            self.graph, mixed
        )["policy_id"]
        index = {
            "status": "complete",
            "runs": [
                {"selected_ids_by_slot": anchor, "policy_id": anchor_id},
                {"selected_ids_by_slot": mixed, "policy_id": mixed_id},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_path = root / "run-index.json"
            confirmation = root / "confirmation.json"
            monitors = [root / f"monitor{number}.json" for number in (1, 2, 3)]
            index_path.write_text(json.dumps(index) + "\n", encoding="utf-8")
            confirmation.write_text("{}\n", encoding="utf-8")
            for monitor in monitors:
                monitor.write_text("{}\n", encoding="utf-8")
            request = {"request_id": "sha256:" + "1" * 64}
            result = screen.build_screen(
                graph=self.graph, request=request, index=index,
                index_path=index_path, controls=self.controls,
                control_summaries=[], confirmation_path=confirmation,
                monitor_paths=monitors, repo_root=Path("/"),
            )
            self.assertEqual(
                {anchor_id, mixed_id}, set(result["policy_cost_by_id"])
            )
            self.assertFalse(
                result["boundary"]["offline_screen_is_runtime_speedup_evidence"]
            )
            self.assertFalse(result["boundary"]["provider_visible"])
            self.assertEqual(
                anchor,
                result["controls"]["anchor"]["selected_ids_by_slot"],
            )
            capability_analysis.verified_screen(
                result, graph=self.graph, request=request, index=index,
                index_path=index_path, repo_root=Path("/"),
            )

    def test_incomplete_catalog_size_and_nonpositive_latency_fail_closed(self):
        missing_arm = copy.deepcopy(self.rows)
        missing_arm[2].pop("flat_double_tree")
        with self.assertRaisesRegex(
            screen.CollectivePolicyScreenError, "incomplete algorithms"
        ):
            screen.analyze_control_rows(self.graph, missing_arm)

        missing_size = copy.deepcopy(self.rows)
        missing_size[1]["baseline_auto"].pop("1024")
        with self.assertRaisesRegex(
            screen.CollectivePolicyScreenError, "incomplete sizes"
        ):
            screen.analyze_control_rows(self.graph, missing_size)

        invalid = copy.deepcopy(self.rows)
        invalid[3]["baseline_auto"]["1024"] = math.nan
        with self.assertRaisesRegex(
            screen.CollectivePolicyScreenError, "invalid latency"
        ):
            screen.analyze_control_rows(self.graph, invalid)

    def test_runner_is_pdebug_only_one_batch_and_three_rotated_blocks(self):
        runner = COLLECTIVE / screen.RUNNER_NAME
        controller = COLLECTIVE / screen.CONTROLLER_NAME
        for script in (runner, controller):
            subprocess.run(["bash", "-n", script], check=True)
            self.assertNotIn("-q pci", script.read_text(encoding="utf-8"))
        controller_text = controller.read_text(encoding="utf-8")
        self.assertEqual(1, controller_text.count("flux batch"))
        self.assertIn("flux batch -q pdebug", controller_text)
        runner_text = runner.read_text(encoding="utf-8")
        self.assertEqual(3, runner_text.count("run_block "))


if __name__ == "__main__":
    unittest.main()
