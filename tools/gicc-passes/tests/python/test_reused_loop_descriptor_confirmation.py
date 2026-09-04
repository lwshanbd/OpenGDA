import subprocess
import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = PASS_ROOT / "experiments" / "reused_loop_descriptor"
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(EXPERIMENT))

import analyze_reused_loop_descriptor_confirmation as confirmation


def rows(value=1.03):
    return {
        allocation: {
            block: {
                batch: {
                    size: value for size in confirmation.SMALL_SIZES
                }
                for batch in confirmation.BATCHES
            }
            for block in confirmation.BLOCKS
        }
        for allocation in confirmation.ALLOCATIONS
    }


class ReusedLoopDescriptorConfirmationTests(unittest.TestCase):
    def test_preregistered_allocation_cluster_gate_passes(self):
        result = confirmation.analyze_speedups(rows())
        self.assertTrue(result["confirmation_gate"]["passed"])
        self.assertEqual(27, result["primary_speedup"]["bootstrap_samples"])
        self.assertEqual(12, result["correctness_gate"]["paired_batch_blocks"])
        self.assertEqual(
            24, result["correctness_gate"]["network_operation_count_audits"],
        )
        self.assertEqual({"4", "64"}, set(result["per_batch"]))

    def test_lower_bound_and_batch_no_regression_are_joint_gates(self):
        losing_allocation = rows(1.10)
        for block in confirmation.BLOCKS:
            for batch in confirmation.BATCHES:
                for size in confirmation.SMALL_SIZES:
                    losing_allocation[3][block][batch][size] = 0.85
        result = confirmation.analyze_speedups(losing_allocation)
        self.assertFalse(result["confirmation_gate"]["passed"])
        self.assertEqual(2, result["primary_speedup"]["allocation_wins"])
        self.assertFalse(
            result["confirmation_gate"]["criteria"]
            ["allocation_cluster_bootstrap_lower_95"]["passed"]
        )

        batch_regression = rows(1.04)
        for allocation in confirmation.ALLOCATIONS:
            for block in confirmation.BLOCKS:
                for size in confirmation.SMALL_SIZES:
                    batch_regression[allocation][block][4][size] = 0.97
        result = confirmation.analyze_speedups(batch_regression)
        self.assertFalse(result["confirmation_gate"]["passed"])
        self.assertFalse(
            result["confirmation_gate"]["criteria"]
            ["both_batch_no_regression_medians"]["passed"]
        )

    def test_invalid_confirmation_shape_fails_closed(self):
        incomplete = rows()
        del incomplete[2]
        with self.assertRaises(confirmation.ConfirmError):
            confirmation.analyze_speedups(incomplete)
        invalid = rows()
        invalid[1][1][4][confirmation.SMALL_SIZES[0]] = 0.0
        with self.assertRaises(confirmation.ConfirmError):
            confirmation.analyze_speedups(invalid)

    def test_driver_order_and_serial_pdebug_entrypoints_are_frozen(self):
        lines = confirmation._expected_driver_lines(1)
        self.assertIn("blocks=AB,BA", lines[0])
        self.assertIn("block=1 order=baseline reused", lines[1])
        self.assertIn("block=2 order=reused baseline", "\n".join(lines))
        scripts = [
            EXPERIMENT / "run_reused_loop_descriptor_confirmation.sh",
            EXPERIMENT / "continue_reused_loop_descriptor_confirmation.sh",
            EXPERIMENT / "continue_reused_loop_descriptor_after_scout.sh",
        ]
        for script in scripts:
            subprocess.run(["bash", "-n", script], check=True)
            text = script.read_text(encoding="utf-8")
            self.assertNotIn("-q pci", text)
            self.assertNotIn("anthropic", text.lower())
        controller = scripts[1].read_text(encoding="utf-8")
        self.assertIn("for allocation in 1 2 3", controller)
        self.assertIn("max_active_or_queued=1", controller)
        self.assertIn("wait_scheduler_idle", controller)
        self.assertIn("flux batch -q pdebug", controller)
        self.assertNotIn("flux cancel", controller)

        successor = scripts[2].read_text(encoding="utf-8")
        self.assertIn("set_state waiting_scout", successor)
        self.assertIn("promising|negative|failed", successor)
        self.assertIn('if [[ $phase != promising ]]', successor)
        self.assertIn('bash "$controller"', successor)
        self.assertNotIn("flux batch", successor)
        self.assertNotIn("flux cancel", successor)


if __name__ == "__main__":
    unittest.main()
