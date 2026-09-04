import copy
import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PASS_ROOT / "experiments"
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(EXPERIMENT_ROOT))

import audit_compiler_llm_readiness as readiness
import gicc_llm_bridge as bridge


def entry(label="unit", graph_id="sha256:" + "1" * 64):
    return {
        "label": label,
        "entry_id": "sha256:" + "2" * 64,
        "graph_id": graph_id,
        "decision_family": "unit-family",
    }


class CompilerLlmReadinessTests(unittest.TestCase):
    def test_failed_placement_gate_is_closed(self):
        summary = {
            "schema_version": "gicc-communication-plan-placement-runtime-v1",
            "model_invoked": False,
            "source_visible_to_model": False,
            "placement_llm_gate": {"passed": False},
        }
        result = readiness.classify_placement(entry(), summary, True)
        self.assertEqual("closed_negative", result["status"])
        self.assertFalse(result["provider_protocol_permitted"])
        self.assertFalse(result["provider_call_authorized"])

    def test_scout_pass_requires_confirmation(self):
        graph_id = "sha256:" + "3" * 64
        payload = {
            "schema_version": "gicc-collective-hierpipe-n8-scout-v1",
            "graph_id": graph_id,
            "model_invoked": False,
            "application_source_modified": False,
            "n8_capacity_gate": {"passed": True},
        }
        analysis = {
            **payload, "result_id": bridge._fingerprint(payload),
        }
        result = readiness.classify_collective(
            entry(graph_id=graph_id), "promising", analysis,
        )
        self.assertEqual("confirmation_required", result["status"])
        self.assertFalse(result["provider_protocol_permitted"])
        self.assertFalse(result["provider_call_authorized"])

    def test_collective_wait_and_negative_paths(self):
        waiting = readiness.classify_collective(entry(), "monitoring", None)
        self.assertEqual("awaiting_scout", waiting["status"])
        graph_id = entry()["graph_id"]
        payload = {
            "schema_version": "gicc-collective-hierpipe-n8-scout-v1",
            "graph_id": graph_id,
            "model_invoked": False,
            "application_source_modified": False,
            "n8_capacity_gate": {"passed": False},
        }
        negative = readiness.classify_collective(
            entry(), "negative",
            {**payload, "result_id": bridge._fingerprint(payload)},
        )
        self.assertEqual("closed_negative", negative["status"])

    def test_producer_scout_pass_never_authorizes_provider(self):
        analysis = {
            "schema_version": "gicc-producer-fission-oracle-analysis-v1",
            "correctness_gate": {"passed": True},
            "oracle_headroom_gate": {"passed": True},
        }
        result = readiness.classify_producer(
            entry(), "promising", analysis,
        )
        self.assertEqual("confirmation_required", result["status"])
        self.assertFalse(result["provider_protocol_permitted"])
        self.assertFalse(result["provider_call_authorized"])

    def test_guarded_scout_waits_and_then_requires_confirmation(self):
        waiting = readiness.classify_guarded_early_trigger(
            entry(), "waiting_predecessor", None,
        )
        self.assertEqual("awaiting_predecessor", waiting["status"])
        analysis = {
            "schema_version": "gicc-guarded-early-trigger-analysis-v1",
            "correctness_gate": {"passed": True},
            "oracle_headroom_gate": {"passed": True},
        }
        result = readiness.classify_guarded_early_trigger(
            entry(), "promising", analysis,
        )
        self.assertEqual("confirmation_required", result["status"])
        self.assertFalse(result["provider_protocol_permitted"])
        self.assertFalse(result["provider_call_authorized"])

    def test_contract_violations_fail_closed(self):
        placement = {
            "schema_version": "gicc-communication-plan-placement-runtime-v1",
            "model_invoked": False,
            "source_visible_to_model": False,
            "placement_llm_gate": {"passed": False},
        }
        leaked = copy.deepcopy(placement)
        leaked["source_visible_to_model"] = True
        with self.assertRaises(readiness.ReadinessError):
            readiness.classify_placement(entry(), leaked, True)
        with self.assertRaises(readiness.ReadinessError):
            readiness.classify_collective(entry(), "promising", None)
        collective_payload = {
            "schema_version": "gicc-collective-hierpipe-n8-scout-v1",
            "graph_id": entry()["graph_id"],
            "model_invoked": False,
            "application_source_modified": False,
            "n8_capacity_gate": {"passed": True},
        }
        with self.assertRaises(readiness.ReadinessError):
            readiness.classify_collective(
                entry(), "negative", {
                    **collective_payload,
                    "result_id": bridge._fingerprint(collective_payload),
                },
            )
        with self.assertRaises(readiness.ReadinessError):
            readiness.classify_producer(
                entry(), "negative", {
                    "schema_version": (
                        "gicc-producer-fission-oracle-analysis-v1"
                    ),
                    "correctness_gate": {"passed": False},
                    "oracle_headroom_gate": {"passed": False},
                },
            )
        with self.assertRaises(readiness.ReadinessError):
            readiness.classify_guarded_early_trigger(
                entry(), "negative", {
                    "schema_version": "gicc-guarded-early-trigger-analysis-v1",
                    "correctness_gate": {"passed": True},
                    "oracle_headroom_gate": {"passed": True},
                },
            )


if __name__ == "__main__":
    unittest.main()
