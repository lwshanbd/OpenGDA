import copy
import importlib.util
import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    PASS_ROOT / "experiments" / "audit_compiler_llm_capability_protocol.py"
)
SPEC = importlib.util.spec_from_file_location("llm_capability_protocol", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def suite_entry():
    return {
        "label": "unit",
        "entry_id": "sha256:" + "1" * 64,
        "graph_id": "sha256:" + "2" * 64,
        "decision_family": "unit-family",
        "views": {
            view: {"prompt_sha256": view + "-sha", "prompt_bytes": 10}
            for view in audit.VIEWS
        },
        "response_schema": {"file_sha256": "schema-sha"},
    }


def readiness_entry(status="awaiting_scout", permitted=False):
    entry = suite_entry()
    return {
        "suite_entry_id": entry["entry_id"],
        "compiler_graph_id": entry["graph_id"],
        "decision_family": entry["decision_family"],
        "status": status,
        "next_stage": "wait",
        "provider_protocol_permitted": permitted,
        "provider_call_authorized": False,
        "llm_performance_measured": False,
    }


def separation_entry():
    entry = suite_entry()
    return {
        "decision_family": entry["decision_family"],
        "same_selectable_ids_across_llm_views": True,
        "views": {
            view: {"prompt": {
                "sha256": entry["views"][view]["prompt_sha256"],
                "bytes": entry["views"][view]["prompt_bytes"],
            }}
            for view in audit.VIEWS
        },
    }


class CompilerLlmCapabilityProtocolTests(unittest.TestCase):
    def test_blocked_entry_permits_no_calls(self):
        result = audit.entry_contract(
            suite_entry(), {"entries": {"unit": readiness_entry()}},
            {"entries": {"unit": separation_entry()}},
        )
        self.assertFalse(result["eligible_to_freeze_provider_request"])
        self.assertEqual(0, result["current_permitted_provider_calls"])
        self.assertEqual(
            60, result["conditional_protocol"]["calls_if_entry_becomes_eligible"]
        )

    def test_runtime_ready_still_requires_separate_authorization(self):
        result = audit.entry_contract(
            suite_entry(),
            {"entries": {"unit": readiness_entry(
                status="provider_protocol_permitted", permitted=True,
            )}},
            {"entries": {"unit": separation_entry()}},
        )
        self.assertTrue(result["eligible_to_freeze_provider_request"])
        self.assertFalse(result["provider_call_authorized"])
        self.assertEqual(0, result["current_permitted_provider_calls"])

    def test_model_invisible_candidate_fails_closed(self):
        ready = readiness_entry(
            status="provider_protocol_permitted", permitted=True,
        )
        ready["candidate_model_visible"] = False
        with self.assertRaisesRegex(
            audit.CapabilityProtocolError, "model-invisible"
        ):
            audit.entry_contract(
                suite_entry(), {"entries": {"unit": ready}},
                {"entries": {"unit": separation_entry()}},
            )

    def test_prompt_or_action_authority_mismatch_fails_closed(self):
        separated = separation_entry()
        separated["same_selectable_ids_across_llm_views"] = False
        with self.assertRaisesRegex(
            audit.CapabilityProtocolError, "unequal authority"
        ):
            audit.entry_contract(
                suite_entry(), {"entries": {"unit": readiness_entry()}},
                {"entries": {"unit": separated}},
            )
        separated = separation_entry()
        separated["views"]["opaque"]["prompt"]["sha256"] = "changed"
        with self.assertRaisesRegex(
            audit.CapabilityProtocolError, "prompt identity mismatch"
        ):
            audit.entry_contract(
                suite_entry(), {"entries": {"unit": readiness_entry()}},
                {"entries": {"unit": separated}},
            )

    def test_content_id_detects_mutation(self):
        payload = {"schema_version": "unit-v1", "value": 1}
        report = {**payload, "id": audit.fingerprint(payload)}
        self.assertEqual(
            report, audit.verified_fingerprint(report, "unit-v1", "id")
        )
        changed = copy.deepcopy(report)
        changed["value"] = 2
        with self.assertRaisesRegex(
            audit.CapabilityProtocolError, "does not match"
        ):
            audit.verified_fingerprint(changed, "unit-v1", "id")

    def test_sampling_null_must_match_suite_policy_count(self):
        entry = suite_entry()
        entry["decision_space"] = {"independent_policy_count": 9}
        suite = {"suite_id": "sha256:" + "3" * 64, "entries": [entry]}
        payload = {
            "schema_version": audit.sampling_null.REPORT_SCHEMA,
            "boundary": dict(audit.sampling_null.BOUNDARY),
            "suite_id": suite["suite_id"],
            "trials_per_view": audit.TRIALS_PER_VIEW,
            "current_suite_uniform_null": {
                "unit": {
                    "legal_policy_count": 9,
                    "draw_count": audit.TRIALS_PER_VIEW,
                    "predesignated_unique_oracle_count": 1,
                },
            },
            "claim_separation": {
                "best_of_20_requires_chance_calibration": True,
                "small_action_spaces_can_hit_oracle_by_chance": True,
                "uniform_null_is_a_model_distribution_claim": False,
                "uniform_null_is_performance_evidence": False,
                "modal_frequency_is_runtime_speedup": False,
                "llm_capability_claim_ready": False,
            },
            "evidence": {"suite": {"sha256": "suite-sha"}},
        }
        report = {
            "null_id": audit.sampling_null.bridge._fingerprint(payload),
            **payload,
        }
        audit.verify_sampling_null(report, suite, "suite-sha")
        changed = copy.deepcopy(report)
        changed["current_suite_uniform_null"]["unit"][
            "legal_policy_count"
        ] = 10
        changed_payload = dict(changed)
        changed_payload.pop("null_id")
        changed["null_id"] = audit.sampling_null.bridge._fingerprint(
            changed_payload
        )
        with self.assertRaisesRegex(
            audit.CapabilityProtocolError, "decision space changed"
        ):
            audit.verify_sampling_null(changed, suite, "suite-sha")


if __name__ == "__main__":
    unittest.main()
