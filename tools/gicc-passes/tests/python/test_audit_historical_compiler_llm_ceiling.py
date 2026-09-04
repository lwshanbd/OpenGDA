import importlib.util
import math
import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    PASS_ROOT / "experiments" / "audit_historical_compiler_llm_ceiling.py"
)
SPEC = importlib.util.spec_from_file_location("historical_llm_ceiling", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def policy(name, frequency, trials):
    return {
        "frequency": frequency,
        "trials": trials,
        "representative_response_sha256": f"response-{name}",
        "actions_by_site": {"site": "proxy"},
    }


def runtime(name, frequency, trials, speedup, lower, upper, p_value, reps=10):
    return {
        "schema_version": audit.RUNTIME_SCHEMA,
        "dossier_id": "dossier",
        "candidate_arm": name,
        "candidate_population_frequency": frequency,
        "candidate_population_trials": trials,
        "replicates": list(range(1, reps + 1)),
        "response_provenance": {
            name: {"response_sha256": f"response-{name}"},
        },
        "binary_sha256_by_arm": {
            "default": "default-binary",
            name: f"binary-{name}",
        },
        "paired_runtime": {
            "default": {
                "replicate_count": reps,
                "speedup_reference_over_candidate_by_replicate": {
                    str(index): speedup for index in range(1, reps + 1)
                },
                "geomean_speedup_reference_over_candidate": speedup,
                "candidate_faster_replicates": reps if speedup > 1 else 0,
                "paired_bootstrap_95_percentile_ci": {
                    "lower_2_5_percent": lower,
                    "upper_97_5_percent": upper,
                    "method": "unit",
                },
                "two_sided_exact_sign_test": {"p_value": p_value},
            },
        },
    }


class HistoricalCompilerLlmCeilingTests(unittest.TestCase):
    def setUp(self):
        # The modal policy is deliberately slower than the one-off policy.
        self.analysis = {
            "dossier_id": "dossier",
            "policies": {
                "llm-policy01": policy("llm-policy01", 15, list(range(1, 16))),
                "llm-policy02": policy("llm-policy02", 5, list(range(16, 21))),
            },
        }
        self.runtime = {
            "llm-policy01": runtime(
                "llm-policy01", 15, list(range(1, 16)), 1.02, 0.99, 1.05, 0.02,
            ),
            "llm-policy02": runtime(
                "llm-policy02", 5, list(range(16, 21)), 1.08, 1.04, 1.11, 0.01,
            ),
        }

    def test_modal_is_frequency_selected_and_best_is_posthoc(self):
        ledgers = {
            name: {"submitted_job_count": 10, "queues": ["pci"]}
            for name in self.runtime
        }
        result = audit.summarize_policies(self.analysis, self.runtime, ledgers)
        self.assertEqual(
            "llm-policy01", result["modal_representative"]["policy"]
        )
        self.assertEqual(
            "llm-policy02",
            result["posthoc_best_of_20_capability_ceiling"]["policy"],
        )
        self.assertTrue(
            result["posthoc_best_of_20_capability_ceiling"]["posthoc"]
        )
        self.assertFalse(
            result["posthoc_best_of_20_capability_ceiling"][
                "statistically_confirmatory"
            ]
        )
        expected = math.exp((15 * math.log(1.02) + 5 * math.log(1.08)) / 20)
        self.assertAlmostEqual(
            expected,
            result["response_population_descriptive"][
                "frequency_weighted_geomean_default_speedup"
            ],
        )
        self.assertFalse(
            result["response_population_descriptive"][
                "paired_confidence_interval_permitted"
            ]
        )

    def test_pci_archive_cannot_support_current_or_stable_claim(self):
        ledgers = {
            name: {"submitted_job_count": 10, "queues": ["pci"]}
            for name in self.runtime
        }
        result = audit.summarize_policies(self.analysis, self.runtime, ledgers)
        self.assertTrue(result["queue_scope"]["historical_only"])
        self.assertEqual(["pci"], result["queue_scope"]["observed_queues"])
        self.assertFalse(
            result["claim_flags"]["current_pdebug_performance_claim_supported"]
        )
        self.assertFalse(
            result["claim_flags"]["stable_modal_speedup_supported"]
        )

    def test_trial_partition_and_route_scope_fail_closed(self):
        ledgers = {
            name: {"submitted_job_count": 10, "queues": ["pci"]}
            for name in self.runtime
        }
        self.analysis["policies"]["llm-policy02"]["trials"][-1] = 19
        with self.assertRaisesRegex(audit.HistoricalCeilingError, "two policies"):
            audit.summarize_policies(self.analysis, self.runtime, ledgers)

        self.setUp()
        self.analysis["policies"]["llm-policy01"]["actions_by_site"]["site"] = (
            "rewrite-source"
        )
        with self.assertRaisesRegex(
            audit.HistoricalCeilingError, "non-route transformation"
        ):
            audit.summarize_policies(self.analysis, self.runtime, ledgers)


if __name__ == "__main__":
    unittest.main()
