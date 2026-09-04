import math
import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PASS_ROOT / "python"))

import gicc_llm_capability_metrics as metrics


ORACLE = {
    "selected_ids_by_slot": {"s0": "a", "s1": "b"},
    "cost_by_unit": {"u0": 1.0, "u1": 4.0},
}
ANCHOR = {
    "selected_ids_by_slot": {"s0": "x", "s1": "y"},
    "cost_by_unit": {"u0": 2.0, "u1": 8.0},
}
DETERMINISTIC = {
    "selected_ids_by_slot": {"s0": "a", "s1": "y"},
    "cost_by_unit": {"u0": 1.5, "u1": 6.0},
}
WEIGHTS = {"u0": 1.0, "u1": 1.0}


def trial(view, index, policy, costs, accepted=True):
    return {
        "view": view,
        "trial": index,
        "bridge_accepted": accepted,
        "selected_ids_by_slot": policy,
        "cost_by_unit": costs,
    }


def scored(view, index, policy, costs, accepted=True):
    return metrics.score_trial(
        trial(view, index, policy, costs, accepted),
        oracle=ORACLE, anchor=ANCHOR, deterministic=DETERMINISTIC,
        unit_weights=WEIGHTS,
    )


class CompilerLlmCapabilityMetricsTests(unittest.TestCase):
    def test_weighted_geomean_and_trial_score(self):
        row = scored(
            "relational", 1, {"s0": "a", "s1": "b"},
            {"u0": 1.0, "u1": 4.0},
        )
        self.assertAlmostEqual(1.0, row["oracle_normalized_geomean_regret"])
        self.assertAlmostEqual(2.0, row["geomean_speedup_over_anchor"])
        self.assertTrue(row["exact_oracle_policy"])
        self.assertEqual(1.0, row["decision_slot_accuracy"])

        ratio = metrics.weighted_geomean_ratio(
            {"a": 4.0, "b": 9.0}, {"a": 1.0, "b": 1.0},
            {"a": 1.0, "b": 1.0},
        )
        self.assertAlmostEqual(6.0, ratio)

    def test_invalid_response_must_use_anchor_fallback(self):
        with self.assertRaisesRegex(
            metrics.CapabilityMetricsError, "anchor fallback"
        ):
            scored(
                "relational", 1, {"s0": "a", "s1": "b"},
                {"u0": 1.0, "u1": 4.0}, accepted=False,
            )
        row = scored(
            "relational", 1, ANCHOR["selected_ids_by_slot"],
            ANCHOR["cost_by_unit"], accepted=False,
        )
        self.assertTrue(row["fallback_applied"])
        self.assertAlmostEqual(2.0, row["oracle_normalized_geomean_regret"])

        malformed_control = {
            "selected_ids_by_slot": {"wrong": "a"},
            "cost_by_unit": DETERMINISTIC["cost_by_unit"],
        }
        with self.assertRaisesRegex(
            metrics.CapabilityMetricsError, "policy slot sets differ"
        ):
            metrics.score_trial(
                trial(
                    "relational", 1, ORACLE["selected_ids_by_slot"],
                    ORACLE["cost_by_unit"],
                ),
                oracle=ORACLE, anchor=ANCHOR, deterministic=malformed_control,
                unit_weights=WEIGHTS,
            )

    def test_modal_tie_is_not_broken_with_oracle_regret(self):
        fast = {"s0": "a", "s1": "b"}
        slow = {"s0": "x", "s1": "b"}
        fast_id = metrics.policy_id(fast)
        slow_id = metrics.policy_id(slow)
        policies = sorted([(fast_id, fast, {"u0": 1.0, "u1": 4.0}),
                           (slow_id, slow, {"u0": 2.0, "u1": 8.0})])
        expected_modal_id = policies[0][0]
        rows = []
        for index in range(1, 21):
            _, policy, costs = policies[index % 2]
            rows.append(scored("relational", index, policy, costs))
        result = metrics.summarize_view(rows)
        self.assertEqual(
            expected_modal_id, result["primary_modal_representative"]["policy_id"]
        )
        self.assertEqual(
            fast_id, result["posthoc_capability_upper_bound"]["policy_id"]
        )
        self.assertEqual(2, result["stability"]["unique_accepted_policy_count"])
        self.assertAlmostEqual(math.log(2), result["stability"][
            "accepted_policy_entropy_nats"
        ])

        inconsistent = [dict(row) for row in rows]
        for row in inconsistent:
            if row["policy_id"] == fast_id:
                row["oracle_normalized_geomean_regret"] = 1.01
                break
        with self.assertRaisesRegex(
            metrics.CapabilityMetricsError, "inconsistent screen costs"
        ):
            metrics.summarize_view(inconsistent)

    def test_three_view_effect_uses_identical_metric_direction(self):
        costs = {
            "relational": {"u0": 1.0, "u1": 4.0},
            "descriptors": {"u0": 1.5, "u1": 6.0},
            "opaque": {"u0": 2.0, "u1": 8.0},
        }
        policy = {"s0": "a", "s1": "b"}
        rows = {
            view: [scored(view, index, policy, cost)
                   for index in range(1, 21)]
            for view, cost in costs.items()
        }
        result = metrics.summarize_views(rows)
        effect = result["semantic_context_effect"]
        self.assertAlmostEqual(
            1.5, effect["descriptors_to_relational_itt_regret_ratio"]
        )
        self.assertAlmostEqual(
            2.0, effect["opaque_to_relational_modal_regret_ratio"]
        )
        self.assertTrue(effect["ratio_above_one_favors_relational"])
        self.assertFalse(effect["performance_superiority_claimed"])

    def test_all_invalid_uses_anchor_as_both_representatives(self):
        rows = [scored(
            "opaque", index, ANCHOR["selected_ids_by_slot"],
            ANCHOR["cost_by_unit"], accepted=False,
        ) for index in range(1, 21)]
        result = metrics.summarize_view(rows)
        self.assertEqual(1.0, result["invalid_output_rate"])
        self.assertEqual(
            "anchor_fallback_no_accepted_response",
            result["primary_modal_representative"]["selection_rule"],
        )
        self.assertEqual(
            result["primary_modal_representative"]["policy_id"],
            result["posthoc_capability_upper_bound"]["policy_id"],
        )


if __name__ == "__main__":
    unittest.main()
