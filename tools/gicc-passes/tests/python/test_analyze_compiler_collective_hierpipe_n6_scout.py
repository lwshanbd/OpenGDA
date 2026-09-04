import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PASS_ROOT / "experiments" / "collective"))

import analyze_compiler_collective_hierpipe_n6_scout as scout


def crossover_rows():
    rows = {}
    for replicate in (1, 2, 3):
        rows[replicate] = {
            scout.base.ARMS[0]: {},
            scout.base.ARMS[1]: {},
            scout.base.ARMS[2]: {},
        }
        for size in scout.base.SIZES:
            key = str(size)
            large = size >= 4194304
            rows[replicate][scout.base.ARMS[0]][key] = (
                300.0 if large else 100.0
            )
            rows[replicate][scout.base.ARMS[1]][key] = (
                100.0 if large else 200.0
            )
            rows[replicate][scout.base.ARMS[2]][key] = 500.0
    return rows


class CompilerCollectiveHierpipeN6ScoutTests(unittest.TestCase):
    def test_n6_wrapper_preserves_gate_and_overrides_runtime_contract(self):
        result = scout.base.analyze_rows(crossover_rows())
        self.assertTrue(result["n6_capacity_gate"]["passed"])
        self.assertEqual(6, scout.base.NODES)
        self.assertEqual(48, scout.base.RANKS)
        self.assertEqual(
            ("analyze_compiler_collective_hierpipe_n8_scout.py",),
            scout.base.SUPPORT_FILENAMES,
        )
        self.assertEqual(
            "gicc-collective-hierpipe-n6-scout-v1",
            scout.base.RESULT_SCHEMA,
        )


if __name__ == "__main__":
    unittest.main()
