import copy
import importlib.util
from fractions import Fraction
import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PASS_ROOT / "experiments" / "audit_llm_sampling_null.py"
SPEC = importlib.util.spec_from_file_location("llm_sampling_null", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


class LlmSamplingNullTests(unittest.TestCase):
    def test_uniform_oracle_probability_is_exact(self):
        result = audit.uniform_oracle_null(5, 20)
        expected = 1 - Fraction(4, 5) ** 20
        self.assertEqual(
            f"{expected.numerator}/{expected.denominator}",
            result["at_least_one_exact_oracle_in_draws_probability"]["exact"],
        )
        self.assertAlmostEqual(
            0.9884707849539315,
            result["at_least_one_exact_oracle_in_draws_probability"]["value"],
        )

    def test_small_and_large_spaces_have_different_nulls(self):
        small = audit.uniform_oracle_null(2, 20)
        large = audit.uniform_oracle_null(4096, 20)
        self.assertGreater(
            small["at_least_one_exact_oracle_in_draws_probability"]["value"],
            0.99999,
        )
        self.assertLess(
            large["at_least_one_exact_oracle_in_draws_probability"]["value"],
            0.005,
        )
        self.assertEqual(15, small["minimum_exact_hits_for_one_sided_alpha_0_05"])
        self.assertEqual(1, large["minimum_exact_hits_for_one_sided_alpha_0_05"])

    def test_historical_occupancy_probabilities(self):
        self.assertAlmostEqual(
            0.9427194306625536,
            float(audit.all_categories_observed_probability(5, 20)),
        )
        self.assertAlmostEqual(
            0.01297411763029932,
            float(audit.maximum_occupancy_tail(5, 20, 10)),
        )

    def test_verifier_rejects_capability_claim(self):
        payload = {
            "schema_version": audit.REPORT_SCHEMA,
            "boundary": dict(audit.BOUNDARY),
            "claim_separation": {
                "best_of_20_requires_chance_calibration": True,
                "small_action_spaces_can_hit_oracle_by_chance": True,
                "uniform_null_is_a_model_distribution_claim": False,
                "uniform_null_is_performance_evidence": False,
                "modal_frequency_is_runtime_speedup": False,
                "llm_capability_claim_ready": False,
            },
        }
        report = {"null_id": audit.bridge._fingerprint(payload), **payload}
        audit.verify_report(report)
        invalid = copy.deepcopy(report)
        invalid["claim_separation"]["llm_capability_claim_ready"] = True
        invalid_payload = dict(invalid)
        invalid_payload.pop("null_id")
        invalid["null_id"] = audit.bridge._fingerprint(invalid_payload)
        with self.assertRaisesRegex(audit.SamplingNullError, "overstates"):
            audit.verify_report(invalid)


if __name__ == "__main__":
    unittest.main()
