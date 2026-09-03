import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PASS_ROOT / "experiments" / "collective"))

import analyze_compiler_collective_hierpipe_n8_scout as scout


def crossover_rows():
    rows = {}
    for replicate in (1, 2, 3):
        rows[replicate] = {
            scout.ARMS[0]: {}, scout.ARMS[1]: {}, scout.ARMS[2]: {},
        }
        for size in scout.SIZES:
            key = str(size)
            large = size >= 4194304
            rows[replicate][scout.ARMS[0]][key] = 300.0 if large else 100.0
            rows[replicate][scout.ARMS[1]][key] = 100.0 if large else 200.0
            rows[replicate][scout.ARMS[2]][key] = 500.0
    return rows


class CompilerCollectiveHierpipeN8ScoutTests(unittest.TestCase):
    def test_stable_small_large_crossover_passes(self):
        result = scout.analyze_rows(crossover_rows())
        self.assertTrue(result["n8_capacity_gate"]["passed"])
        self.assertEqual(
            [scout.ARMS[0], scout.ARMS[1]],
            result["aggregate"]["distinct_size_winners"],
        )
        self.assertGreater(
            result["aggregate"]["best_uniform_over_pointwise_geomean"],
            1.05,
        )
        for row in result["pipeline_winner_persistence"].values():
            self.assertEqual(3, row["wins_over_best_uniform"])

    def test_uniform_winner_fails_capacity_gate(self):
        rows = crossover_rows()
        for replicate in rows:
            for size in scout.SIZES:
                key = str(size)
                rows[replicate][scout.ARMS[0]][key] = 100.0
                rows[replicate][scout.ARMS[1]][key] = 200.0
                rows[replicate][scout.ARMS[2]][key] = 300.0
        result = scout.analyze_rows(rows)
        self.assertFalse(result["n8_capacity_gate"]["passed"])
        self.assertEqual([scout.ARMS[0]], result["aggregate"]["distinct_size_winners"])

    def test_pooled_pipeline_outlier_must_persist_in_paired_blocks(self):
        rows = crossover_rows()
        key = "4194304"
        for replicate in rows:
            for other in ("8388608", "16777216"):
                rows[replicate][scout.ARMS[1]][other] = 400.0
        rows[1][scout.ARMS[0]][key] = 1000.0
        rows[1][scout.ARMS[1]][key] = 1.0
        rows[2][scout.ARMS[0]][key] = 2.0
        rows[2][scout.ARMS[1]][key] = 3.0
        rows[3][scout.ARMS[0]][key] = 100.0
        rows[3][scout.ARMS[1]][key] = 101.0
        result = scout.analyze_rows(rows)
        self.assertEqual(scout.ARMS[1], result["per_size_pooled_median"][key]["winner"])
        self.assertEqual(
            1,
            result["pipeline_winner_persistence"][key]["wins_over_best_uniform"],
        )
        self.assertFalse(result["n8_capacity_gate"]["passed"])

    def test_requires_three_complete_blocks(self):
        with self.assertRaisesRegex(scout.ScoutError, "blocks 1, 2, and 3"):
            scout.analyze_rows({1: crossover_rows()[1]})


if __name__ == "__main__":
    unittest.main()
