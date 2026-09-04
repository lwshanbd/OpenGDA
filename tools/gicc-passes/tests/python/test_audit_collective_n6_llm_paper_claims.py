import importlib.util
import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = PASS_ROOT / "experiments"
COLLECTIVE = EXPERIMENTS / "collective"
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(COLLECTIVE))
SCRIPT = EXPERIMENTS / "audit_collective_n6_llm_paper_claims.py"
SPEC = importlib.util.spec_from_file_location("n6_paper_claims", SCRIPT)
claims = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = claims
SPEC.loader.exec_module(claims)


def evidence(stable=True, context=True, ceiling=True):
    capability = {
        "label": "collective_n6",
        "analysis_id": "analysis",
        "screen_id": "screen",
        "compiler_graph_id": "graph",
        "decision_family": "collective_algorithm_and_size_policy",
        "boundary": {
            "compiler_lto_decisions_only": True,
            "application_source_visible_or_modified": False,
        },
    }
    distance = {
        "subject_cost_regret_to_oracle": {"estimate": 1.07},
        "used_as_a_pass_fail_gate": False,
    }
    runtime = {
        "plan_id": "plan",
        "compiler_graph_id": "graph",
        "boundary": {
            "compiler_lto_decisions_only": True,
            "application_source_modified": False,
            "runtime_labels_visible_to_model": False,
        },
        "claim_gates": {
            "stable_relational_modal_runtime_improvement_supported": stable,
            "relational_context_runtime_effect_supported": context,
            "posthoc_capability_ceiling_runtime_potential_observed": ceiling,
        },
        "representative_comparisons": {
            "relational:primary_modal_representative": {
                "distance_to_runtime_oracle": distance,
            },
        },
    }
    plan = {
        "plan_id": "plan",
        "capability_analysis_id": "analysis",
        "policy_screen_id": "screen",
    }
    suite = {
        "entries": [
            {
                "label": "collective_n6",
                "decision_family": "collective_algorithm_and_size_policy",
                "entry_id": "coll-entry", "graph_id": "graph",
                "decision_space": {"independent_policy_count": 4096},
            },
            {
                "label": "jacobi",
                "decision_family": "communication_route_or_schedule",
                "entry_id": "jacobi-entry", "graph_id": "jacobi-graph",
                "decision_space": {"independent_policy_count": 9},
            },
            {
                "label": "loop_lto",
                "decision_family": "communication_route_or_schedule",
                "entry_id": "loop-entry", "graph_id": "loop-graph",
                "decision_space": {"independent_policy_count": 2},
            },
        ],
    }
    protocol = {
        "entries": {
            "collective_n6": {
                "eligible_to_freeze_provider_request": True,
            },
            "jacobi": {"eligible_to_freeze_provider_request": True},
            "loop_lto": {"eligible_to_freeze_provider_request": False},
        },
    }
    return capability, runtime, plan, suite, protocol, distance


class AuditCollectiveN6LlmPaperClaimsTests(unittest.TestCase):
    def test_positive_single_family_result_does_not_claim_generalization(self):
        capability, runtime, plan, suite, protocol, distance = evidence()
        result = claims.claim_summary(
            capability, runtime, plan, suite, protocol
        )
        self.assertEqual(
            "single_family_relational_effect_supported", result["status"]
        )
        self.assertTrue(result["claim_matrix"][
            "stable_relational_modal_runtime_improvement_supported"
        ])
        self.assertFalse(
            result["claim_matrix"]["portfolio_generalization_supported"]
        )
        self.assertFalse(result["claim_matrix"]["paper_mainline_complete"])
        self.assertEqual(
            distance, result["relational_modal_runtime_distance_to_oracle"]
        )
        self.assertEqual(
            ["jacobi"],
            [entry["label"] for entry in result[
                "different_family_followup"
            ]["currently_eligible"]],
        )
        self.assertFalse(
            result["different_family_followup"]["automatically_authorized"]
        )

    def test_posthoc_ceiling_is_not_promoted_to_stable_policy(self):
        capability, runtime, plan, suite, protocol, _ = evidence(
            stable=False, context=False, ceiling=True
        )
        result = claims.claim_summary(
            capability, runtime, plan, suite, protocol
        )
        self.assertEqual("posthoc_capability_ceiling_only", result["status"])
        self.assertFalse(result["claim_matrix"][
            "stable_relational_modal_runtime_improvement_supported"
        ])
        self.assertTrue(result["interpretation_constraints"][
            "best_of_20_is_posthoc_only"
        ])

    def test_same_family_entries_cannot_satisfy_followup(self):
        capability, runtime, plan, suite, protocol, _ = evidence()
        suite["entries"][1]["decision_family"] = (
            "collective_algorithm_and_size_policy"
        )
        result = claims.claim_summary(
            capability, runtime, plan, suite, protocol
        )
        self.assertFalse(result["different_family_followup"]["available"])
        self.assertEqual(
            "establish_stable_compiler_oracle_in_another_family",
            result["different_family_followup"]["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
