import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = (
    ROOT / "tools" / "gicc-passes" / "experiments" / "guarded_early_trigger"
)
sys.path.insert(0, str(EXPERIMENT))

import audit_partial_negative_scout as audit


class GuardedEarlyPartialNegativeTests(unittest.TestCase):
    def test_completed_parser_accepts_scientific_timings(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            stdout = directory / "run.out"
            stderr = directory / "run.err"
            lines = [
                f"Run {run}: {3.0e6 + run:.5e} us" +
                (" (warmup)" if run < 2 else "")
                for run in range(10)
            ]
            lines.append("gicc::launch average (runs 2-9): 3.00001e+06 us")
            stdout.write_text("\n".join(lines) + "\n", encoding="utf-8")
            expected_bytes = 4096 * (4096 // 16) * 4
            stderr.write_text("\n".join(
                f"GICC_MM_CHECKSUM rank={rank} bytes={expected_bytes} "
                f"fnv64={rank:016x} launches=161"
                for rank in range(16)
            ) + "\n", encoding="utf-8")
            result = audit.parse_completed_run(
                stdout, stderr, 4096, "baseline")
        self.assertEqual(10, len(result["run_us"]))
        self.assertEqual(16, len(result["checksums"]))

    def test_missing_pair_cannot_rescue_observed_one_win(self):
        result = audit.gate_reachability([0.9977, 1.0046, 0.9979])
        self.assertEqual(2, result["maximum_possible_wins"])
        self.assertLess(result["maximum_possible_median_speedup"], 1.02)
        self.assertFalse(
            result["gate_reachable_under_arbitrarily_favorable_missing_pair"]
        )

    def test_missing_pair_can_rescue_two_wins_above_threshold(self):
        result = audit.gate_reachability([1.03, 1.04, 0.99])
        self.assertEqual(3, result["maximum_possible_wins"])
        self.assertGreaterEqual(result["maximum_possible_median_speedup"], 1.02)
        self.assertTrue(
            result["gate_reachable_under_arbitrarily_favorable_missing_pair"]
        )


if __name__ == "__main__":
    unittest.main()
