import copy
import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
COLLECTIVE = PASS_ROOT / "experiments" / "collective"
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(COLLECTIVE))

import analyze_compiler_collective_hierpipe_n8_scout as scout
import gicc_llm_bridge as bridge
import prepare_compiler_collective_n8_confirmation as transition


def crossover_rows():
    rows = {}
    for replicate in (1, 2, 3):
        rows[replicate] = {arm: {} for arm in scout.ARMS}
        for size in scout.SIZES:
            key = str(size)
            large = size >= 4194304
            rows[replicate][scout.ARMS[0]][key] = (
                300.0 if large else 100.0
            )
            rows[replicate][scout.ARMS[1]][key] = (
                100.0 if large else 200.0
            )
            rows[replicate][scout.ARMS[2]][key] = 500.0
    return rows


def passed_scout():
    analyzed = scout.analyze_rows(crossover_rows())
    payload = {
        "schema_version": "gicc-collective-hierpipe-n8-scout-v1",
        "scope": "unit compiler-only scout",
        "model_invoked": False,
        "application_source_modified": False,
        "graph_id": transition.GRAPH_ID,
        "bundle_id": transition.BUNDLE_ID,
        "job_id": "unit-job",
        "nodelist": ["unit0", "unit1"],
        "block_monitors": [],
        **analyzed,
    }
    return {**payload, "result_id": bridge._fingerprint(payload)}


class CompilerCollectiveN8ConfirmationTransitionTests(unittest.TestCase):
    def test_passed_scout_maps_to_fixed_compiler_bins(self):
        selected, bins, uniform = transition.derive_bin_algorithms(
            passed_scout()
        )
        self.assertEqual(
            {
                "message-bin-0": scout.ARMS[0],
                "message-bin-1": scout.ARMS[0],
                "message-bin-2": scout.ARMS[1],
                "message-bin-3": scout.ARMS[1],
            },
            selected,
        )
        self.assertEqual(scout.ARMS[0], uniform)
        self.assertEqual(
            [1048576, 4194304, 8388608],
            bins["message-bin-2"]["scout_sizes"],
        )

    def test_tie_break_is_preregistered(self):
        value = passed_scout()
        for row in value["per_size_pooled_median"].values():
            for arm in scout.ARMS:
                row["algorithm_median_us"][arm] = 100.0
            row["winner"] = scout.ARMS[0]
        values = {arm: 100.0 for arm in scout.ARMS}
        value["aggregate"]["algorithm_geomean_us"] = values
        value["aggregate"]["best_uniform_algorithm"] = scout.ARMS[0]
        payload = dict(value)
        payload.pop("result_id")
        value["result_id"] = bridge._fingerprint(payload)
        selected, _, uniform = transition.derive_bin_algorithms(value)
        self.assertEqual({scout.ARMS[0]}, set(selected.values()))
        self.assertEqual(scout.ARMS[0], uniform)

    def test_negative_or_tampered_scout_is_rejected(self):
        negative = passed_scout()
        negative["n8_capacity_gate"]["passed"] = False
        payload = dict(negative)
        payload.pop("result_id")
        negative["result_id"] = bridge._fingerprint(payload)
        with self.assertRaises(transition.TransitionError):
            transition.derive_bin_algorithms(negative)

        tampered = passed_scout()
        tampered["per_size_pooled_median"]["1024"][
            "algorithm_median_us"
        ][scout.ARMS[0]] = 0.0
        with self.assertRaises(transition.TransitionError):
            transition.derive_bin_algorithms(tampered)

    def test_policy_equal_to_a_simple_control_has_no_incremental_surface(self):
        derived = {"b0": "a", "b1": "b"}
        self.assertFalse(transition.has_incremental_policy(
            derived, dict(derived), {"b0": "a", "b1": "c"},
        ))
        self.assertFalse(transition.has_incremental_policy(
            derived, {"b0": "a", "b1": "c"}, dict(derived),
        ))
        self.assertTrue(transition.has_incremental_policy(
            derived,
            {"b0": "a", "b1": "c"},
            {"b0": "d", "b1": "b"},
        ))

    def test_reported_winner_and_uniform_must_regenerate(self):
        wrong_winner = passed_scout()
        wrong_winner["per_size_pooled_median"]["1024"]["winner"] = (
            scout.ARMS[2]
        )
        payload = dict(wrong_winner)
        payload.pop("result_id")
        wrong_winner["result_id"] = bridge._fingerprint(payload)
        with self.assertRaisesRegex(
            transition.TransitionError, "winner does not match"
        ):
            transition.derive_bin_algorithms(wrong_winner)

        wrong_uniform = passed_scout()
        wrong_uniform["aggregate"]["best_uniform_algorithm"] = scout.ARMS[2]
        payload = copy.deepcopy(wrong_uniform)
        payload.pop("result_id")
        wrong_uniform["result_id"] = bridge._fingerprint(payload)
        with self.assertRaisesRegex(
            transition.TransitionError, "best-uniform"
        ):
            transition.derive_bin_algorithms(wrong_uniform)


if __name__ == "__main__":
    unittest.main()
