import copy
import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = PASS_ROOT / "experiments" / "producer_fission"
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(EXPERIMENT))

import prepare_producer_fission_confirmation as transition


def monitor_value(first=(1.04, 1.06, 1.05, 0.99),
                  second=(0.98, 0.99, 1.00, 0.97)):
    value = {
        "schema_version": transition.MONITOR_SCHEMA,
        "state": "passed",
        "expected": {
            "queue": "pdebug", "nodes": 2, "ranks": 16, "ppn": 8,
            "replicates": 4, "sizes": [1024, 4096], "iterations": 200,
            "nccheck": 10,
        },
        "job_id": "unit-job",
        "scheduler": {"exit_code": 0, "exception_types": []},
        "jobspec": {
            "queue": "pdebug",
            "duration_seconds": 1200.0,
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
                "baseline": {
                    "seconds": speedup,
                    "iterations": 200,
                    "final_l2": 1e-4,
                },
                "fission": {
                    "seconds": 1.0,
                    "iterations": 200,
                    "final_l2": 1e-4,
                },
                "speedup": speedup,
            }
    return value


class ProducerFissionConfirmationTransitionTests(unittest.TestCase):
    def test_recompute_retains_all_sizes_after_one_size_passes(self):
        result = transition.recompute_scout(monitor_value())
        self.assertTrue(result["oracle_headroom_gate"]["passed"])
        self.assertEqual(
            [1024], result["oracle_headroom_gate"]["promising_sizes"]
        )
        self.assertEqual(
            {"1024", "4096"}, set(result["sizes"])
        )

    def test_analysis_must_regenerate_from_runtime_pairs(self):
        monitor = monitor_value()
        result = transition.recompute_scout(monitor)
        analysis = {
            "schema_version": transition.ANALYSIS_SCHEMA,
            "created_at": "unit",
            "monitor": "/tmp/unit-monitor.json",
            **result,
        }
        validated = transition.validate_analysis(
            analysis, monitor, Path("/tmp/unit-monitor.json")
        )
        self.assertIs(validated, analysis)
        tampered = copy.deepcopy(analysis)
        tampered["sizes"]["1024"]["median_speedup"] = 9.0
        with self.assertRaisesRegex(
            transition.TransitionError, "does not regenerate"
        ):
            transition.validate_analysis(
                tampered, monitor, Path("/tmp/unit-monitor.json")
            )

    def test_runtime_contract_is_exactly_pdebug_n2(self):
        monitor = monitor_value()
        transition.validate_runtime_contract(monitor)
        monitor["jobspec"]["queue"] = "pci"
        with self.assertRaisesRegex(transition.TransitionError, "Flux contract"):
            transition.validate_runtime_contract(monitor)

    def test_negative_scout_cannot_enter_confirmation(self):
        monitor = monitor_value(
            first=(0.99, 0.98, 0.97, 0.96),
            second=(0.98, 0.99, 1.00, 0.97),
        )
        result = transition.recompute_scout(monitor)
        self.assertFalse(result["oracle_headroom_gate"]["passed"])
        analysis = {
            "schema_version": transition.ANALYSIS_SCHEMA,
            "created_at": "unit",
            "monitor": "/tmp/unit-monitor.json",
            **result,
        }
        with self.assertRaisesRegex(transition.TransitionError, "did not pass"):
            transition.validate_analysis(
                analysis, monitor, Path("/tmp/unit-monitor.json")
            )

    def test_tampered_speedup_or_correctness_is_rejected(self):
        monitor = monitor_value()
        monitor["runs"]["1"]["1024"]["speedup"] = 12.0
        with self.assertRaisesRegex(
            transition.TransitionError, "does not regenerate"
        ):
            transition.recompute_scout(monitor)
        monitor = monitor_value()
        monitor["runs"]["1"]["1024"]["fission"]["final_l2"] = 2e-4
        with self.assertRaisesRegex(
            transition.TransitionError, "correctness"
        ):
            transition.recompute_scout(monitor)


if __name__ == "__main__":
    unittest.main()
