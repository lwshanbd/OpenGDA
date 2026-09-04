import copy
import subprocess
import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = PASS_ROOT / "experiments" / "guarded_early_trigger"
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(EXPERIMENT))

import analyze_guarded_early_trigger_confirmation as confirmation
import prepare_guarded_early_trigger_confirmation as transition


def arm(runtime, launches):
    return {
        "measured_mean_us": runtime,
        "checksums": {str(rank): f"{rank:016x}" for rank in range(16)},
        "successful_kernel_launches": {
            str(rank): launches for rank in range(16)
        },
    }


def scout_value(first=(1.03, 1.04, 1.05, 0.99), second=(0.98,) * 4):
    value = {
        "schema_version": transition.MONITOR_SCHEMA,
        "state": "passed",
        "expected": {
            "queue": "pdebug", "nodes": 2, "ranks": 16, "ppn": 8,
            "replicates": 4, "sizes": [4096, 8192],
            "application_runs": 10, "application_warmup": 2,
        },
        "job_id": "unit-job",
        "scheduler": {"exit_code": 0, "exception_types": []},
        "jobspec": {
            "queue": "pdebug", "duration_seconds": 1800.0,
            "resources": [{
                "type": "node", "count": 2,
                "with": [{
                    "type": "slot", "count": 8, "label": "task",
                    "with": [
                        {"type": "core", "count": 8},
                        {"type": "gpu", "count": 1},
                    ],
                }],
            }],
        },
        "resource_set": {"nodelist": ["unit0", "unit1"]},
        "runs": {},
    }
    for replicate, pair in enumerate(zip(first, second), 1):
        value["runs"][str(replicate)] = {}
        for size, speedup in zip(transition.SIZES, pair):
            value["runs"][str(replicate)][str(size)] = {
                "baseline": arm(speedup, 161),
                "guarded": arm(1.0, 483),
                "speedup": speedup,
            }
    return value


def rows(value=1.04):
    return {
        allocation: {
            block: {str(size): value for size in confirmation.SIZES}
            for block in confirmation.BLOCKS
        }
        for allocation in confirmation.ALLOCATIONS
    }


class GuardedEarlyTriggerConfirmationTests(unittest.TestCase):
    def test_scout_recomputation_retains_both_sizes(self):
        result = transition.recompute_scout(scout_value())
        self.assertTrue(result["oracle_headroom_gate"]["passed"])
        self.assertEqual([4096], result["oracle_headroom_gate"]["promising_sizes"])
        self.assertEqual({"4096", "8192"}, set(result["sizes"]))

    def test_scout_rejects_silent_fallback_and_checksum_change(self):
        value = scout_value()
        value["runs"]["1"]["4096"]["guarded"][
            "successful_kernel_launches"
        ]["0"] = 161
        with self.assertRaisesRegex(transition.TransitionError, "attestation"):
            transition.recompute_scout(value)
        value = scout_value()
        value["runs"]["1"]["4096"]["guarded"]["checksums"]["0"] = "f" * 16
        with self.assertRaisesRegex(transition.TransitionError, "correctness"):
            transition.recompute_scout(value)

    def test_runtime_contract_is_exact_n2_pdebug(self):
        value = scout_value()
        transition.validate_runtime_contract(value)
        value["jobspec"]["queue"] = "pci"
        with self.assertRaisesRegex(transition.TransitionError, "Flux contract"):
            transition.validate_runtime_contract(value)

    def test_confirmation_gate_is_allocation_clustered(self):
        result = confirmation.analyze_speedups(rows())
        self.assertTrue(result["confirmation_gate"]["passed"])
        self.assertEqual(27, result["primary_speedup"]["bootstrap_samples"])
        self.assertEqual(12, result["runtime_guard_gate"]["paired_checks"])
        losing = rows(1.10)
        for block in confirmation.BLOCKS:
            for size in confirmation.SIZES:
                losing[3][block][str(size)] = 0.85
        self.assertFalse(
            confirmation.analyze_speedups(losing)["confirmation_gate"]["passed"]
        )

    def test_invalid_confirmation_shape_fails_closed(self):
        value = rows()
        del value[2]
        with self.assertRaises(confirmation.ConfirmError):
            confirmation.analyze_speedups(value)
        value = rows()
        value[1][1]["4096"] = 0.0
        with self.assertRaises(confirmation.ConfirmError):
            confirmation.analyze_speedups(value)

    def test_serial_entrypoints_and_order_are_frozen(self):
        lines = confirmation._expected_driver_lines(1)
        self.assertIn("blocks=AB,BA", lines[0])
        self.assertIn("block=1 order=baseline guarded", lines[1])
        self.assertIn("block=2 order=guarded baseline", "\n".join(lines))
        scripts = [
            EXPERIMENT / "run_guarded_early_trigger_confirmation.sh",
            EXPERIMENT / "continue_guarded_early_trigger_confirmation.sh",
            EXPERIMENT / "continue_guarded_early_trigger_after_scout.sh",
        ]
        for script in scripts:
            subprocess.run(["bash", "-n", script], check=True)
            text = script.read_text(encoding="utf-8")
            self.assertNotIn("-q pci", text)
            self.assertNotIn("anthropic", text.lower())
        controller = scripts[1].read_text(encoding="utf-8")
        self.assertIn("for allocation in 1 2 3", controller)
        self.assertIn("max_active_or_queued=1", controller)


if __name__ == "__main__":
    unittest.main()
