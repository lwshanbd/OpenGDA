import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PASS_ROOT / "experiments" / "collective"))

import analyze_compiler_collective_cross_topology as cross


def n2_input():
    return {
        "schema_version": "gicc-collective-control-analysis-v1",
        "gate_c": {"passed": False},
        "per_size": {
            str(size): {
                "algorithm_median_us": {
                    "baseline_auto": 100.0,
                    "hierarchical_double_tree": 200.0,
                },
            }
            for size in cross.SIZES
        },
    }


def n4_input():
    return {
        "schema_version": "gicc-collective-topology-scout-analysis-v1",
        "model_invoked": False,
        "application_source_modified": False,
        "v4_topology_hypothesis": {"promising": False},
        "per_size": {
            str(size): {
                "baseline_us": 300.0,
                "hierarchical_double_tree_us": 100.0,
            }
            for size in cross.SIZES
        },
    }


class CompilerCollectiveCrossTopologyTests(unittest.TestCase):
    def test_topology_crossover_has_space_but_not_unique_llm_value(self):
        result = cross.analyze_values(n2_input(), n4_input())
        diagnosis = result["decision_diagnosis"]
        self.assertEqual(
            {"2": "baseline_auto", "4": "hierarchical_double_tree"},
            diagnosis["observed_topology_rule"],
        )
        self.assertTrue(diagnosis["winner_changes_with_nodes"])
        self.assertTrue(diagnosis["all_18_point_winners_match_topology_rule"])
        self.assertFalse(diagnosis["llm_unique_value_demonstrated"])
        aggregate = result["equal_topology_weight_aggregate"]
        self.assertEqual(
            "hierarchical_double_tree", aggregate["best_uniform_algorithm"],
        )
        self.assertGreater(
            aggregate["best_uniform_over_topology_conditional"], 1.2,
        )

    def test_closed_inputs_and_complete_sizes_are_required(self):
        invalid_n2 = n2_input()
        invalid_n2["gate_c"]["passed"] = True
        with self.assertRaisesRegex(cross.CrossTopologyError, "negative v3"):
            cross.analyze_values(invalid_n2, n4_input())

        invalid_n4 = n4_input()
        del invalid_n4["per_size"][str(cross.SIZES[0])]
        with self.assertRaisesRegex(cross.CrossTopologyError, "size coverage"):
            cross.analyze_values(n2_input(), invalid_n4)


if __name__ == "__main__":
    unittest.main()
