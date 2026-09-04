import subprocess
import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = PASS_ROOT / "experiments" / "producer_fission"
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(EXPERIMENT))

import analyze_producer_fission_confirmation as confirmation


def rows(value=1.06):
    return {
        allocation: {
            block: {str(size): value for size in confirmation.SIZES}
            for block in confirmation.BLOCKS
        }
        for allocation in confirmation.ALLOCATIONS
    }


class ProducerFissionConfirmationTests(unittest.TestCase):
    def test_preregistered_aggregate_gate_passes(self):
        result = confirmation.analyze_speedups(rows())
        self.assertTrue(result["confirmation_gate"]["passed"])
        self.assertEqual(12, result["correctness_gate"]["paired_checks"])
        self.assertEqual(
            {"1024", "4096"}, set(result["per_size"])
        )
        self.assertAlmostEqual(1.06, result["primary_speedup"]["estimate"])

    def test_point_estimate_lower_bound_and_allocation_wins_are_joint(self):
        values = rows(1.20)
        for block in confirmation.BLOCKS:
            for size in confirmation.SIZES:
                values[3][block][str(size)] = 0.85
        result = confirmation.analyze_speedups(values)
        self.assertEqual(2, result["primary_speedup"]["allocation_wins"])
        self.assertFalse(result["confirmation_gate"]["passed"])
        self.assertFalse(
            result["confirmation_gate"]["criteria"]
            ["allocation_cluster_bootstrap_lower_95"]["passed"]
        )

    def test_bootstrap_clusters_allocations(self):
        result = confirmation.allocation_bootstrap([1.04, 1.05, 1.06])
        self.assertEqual(27, result["bootstrap_samples"])
        self.assertEqual(3, result["allocation_wins"])
        self.assertIn("clustered", result["method"])

    def test_missing_or_invalid_data_is_rejected(self):
        incomplete = rows()
        del incomplete[2]
        with self.assertRaises(confirmation.ConfirmError):
            confirmation.analyze_speedups(incomplete)
        invalid = rows()
        invalid[1][1]["1024"] = 0.0
        with self.assertRaises(confirmation.ConfirmError):
            confirmation.analyze_speedups(invalid)

    def test_driver_order_and_shell_entrypoints_are_frozen(self):
        lines = confirmation._expected_driver_lines(1)
        self.assertIn("blocks=AB,BA", lines[0])
        self.assertIn("block=1 order=baseline fission", lines[1])
        self.assertIn("block=2 order=fission baseline", "\n".join(lines))
        scripts = [
            EXPERIMENT / "run_producer_fission_confirmation.sh",
            EXPERIMENT / "continue_producer_fission_confirmation.sh",
        ]
        for script in scripts:
            subprocess.run(["bash", "-n", script], check=True)
            self.assertNotIn("-q pci", script.read_text(encoding="utf-8"))
        controller = scripts[1].read_text(encoding="utf-8")
        self.assertIn("-q pdebug", controller)
        self.assertIn("for allocation in 1 2 3", controller)
        self.assertIn('python3 "$monitor"', controller)


if __name__ == "__main__":
    unittest.main()
