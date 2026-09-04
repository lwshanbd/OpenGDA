import subprocess
import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
COLLECTIVE = PASS_ROOT / "experiments" / "collective"
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(COLLECTIVE))

import analyze_compiler_collective_n8_confirmation as confirmation


def rows(derived=100.0, uniform=110.0, heuristic=106.0):
    costs = {
        confirmation.ARMS[0]: derived,
        confirmation.ARMS[1]: uniform,
        confirmation.ARMS[2]: heuristic,
    }
    return {
        replicate: {
            arm: {str(size): value for size in confirmation.SIZES}
            for arm, value in costs.items()
        }
        for replicate in (1, 2, 3)
    }


class CompilerCollectiveN8ConfirmationTests(unittest.TestCase):
    def test_both_preregistered_comparators_must_pass(self):
        result = confirmation.analyze_rows(rows())
        self.assertTrue(result["confirmation_gate"]["passed"])
        self.assertAlmostEqual(
            1.10,
            result["comparisons"]["scout_best_uniform"]
            ["derived_policy_speedup"]["estimate"],
        )
        self.assertAlmostEqual(
            1.06,
            result["comparisons"]["frozen_structural_heuristic"]
            ["derived_policy_speedup"]["lower_2_5_percent"],
        )

        weak = confirmation.analyze_rows(rows(heuristic=102.0))
        self.assertFalse(weak["confirmation_gate"]["passed"])
        self.assertTrue(
            weak["confirmation_gate"]["criteria"]
            ["derived_over_scout_best_uniform_point_estimate"]["passed"]
        )
        self.assertFalse(
            weak["confirmation_gate"]["criteria"]
            ["derived_over_frozen_structural_heuristic_point_estimate"]
            ["passed"]
        )

    def test_bootstrap_clusters_three_independent_allocations(self):
        result = confirmation.paired_allocation_bootstrap([1.04, 1.06, 1.08])
        self.assertEqual(27, result["bootstrap_samples"])
        self.assertEqual(3, result["allocation_wins"])
        self.assertIn("independent pdebug allocations", result["method"])

    def test_incomplete_or_nonpositive_rows_are_rejected(self):
        incomplete = rows()
        del incomplete[3]
        with self.assertRaises(confirmation.ConfirmError):
            confirmation.analyze_rows(incomplete)
        invalid = rows()
        invalid[2][confirmation.ARMS[0]][str(confirmation.SIZES[0])] = 0.0
        with self.assertRaises(confirmation.ConfirmError):
            confirmation.analyze_rows(invalid)

    def test_shell_entrypoints_are_syntax_checked_and_pdebug_only(self):
        scripts = [
            COLLECTIVE / "run_compiler_collective_n8_confirmation.sh",
            COLLECTIVE / "continue_compiler_collective_n8_confirmation.sh",
        ]
        for script in scripts:
            subprocess.run(["bash", "-n", script], check=True)
            text = script.read_text(encoding="utf-8")
            self.assertNotIn("-q pci", text)
        controller = scripts[1].read_text(encoding="utf-8")
        self.assertIn("-q pdebug", controller)
        self.assertIn("for replicate in 1 2 3", controller)
        self.assertIn('python3 "$monitor"', controller)


if __name__ == "__main__":
    unittest.main()
