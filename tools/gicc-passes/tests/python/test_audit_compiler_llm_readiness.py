import copy
import json
import sys
import tempfile
import unittest
from unittest import mock
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

    def test_passed_collective_confirmation_permits_request_not_provider(self):
        graph_id = "sha256:" + "3" * 64
        payload = {
            "schema_version": "gicc-collective-hierpipe-n8-scout-v1",
            "graph_id": graph_id,
            "model_invoked": False,
            "application_source_modified": False,
            "n8_capacity_gate": {"passed": True},
        }
        analysis = {**payload, "result_id": bridge._fingerprint(payload)}
        result = readiness.classify_collective(
            entry(graph_id=graph_id), "promising", analysis,
            "confirmed", True,
        )
        self.assertEqual("provider_protocol_permitted", result["status"])
        self.assertTrue(result["provider_protocol_permitted"])
        self.assertTrue(result["runtime_confirmation_gate_passed"])
        self.assertFalse(result["provider_call_authorized"])

        negative = readiness.classify_collective(
            entry(graph_id=graph_id), "promising", analysis,
            "negative", False,
        )
        self.assertEqual("closed_negative", negative["status"])
        self.assertFalse(negative["provider_protocol_permitted"])

    def test_collective_confirmation_state_must_match_result(self):
        graph_id = "sha256:" + "3" * 64
        payload = {
            "schema_version": "gicc-collective-hierpipe-n8-scout-v1",
            "graph_id": graph_id,
            "model_invoked": False,
            "application_source_modified": False,
            "n8_capacity_gate": {"passed": True},
        }
        analysis = {**payload, "result_id": bridge._fingerprint(payload)}
        with self.assertRaisesRegex(
            readiness.ReadinessError, "confirmation state disagrees"
        ):
            readiness.classify_collective(
                entry(graph_id=graph_id), "promising", analysis,
                "negative", True,
            )

    def test_collective_confirmation_replays_before_eligibility(self):
        graph_id = "sha256:" + "4" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transition = root / "transition.json"
            graph = root / "graph.json"
            transition.write_text("{}\n", encoding="utf-8")
            graph.write_text(
                json.dumps({"graph_id": graph_id}) + "\n",
                encoding="utf-8",
            )
            monitors = [root / f"monitor{number}.json" for number in (1, 2, 3)]
            for path in monitors:
                path.write_text("{}\n", encoding="utf-8")
            payload = {
                "schema_version": "gicc-collective-n8-confirmation-v1",
                "model_invoked": False,
                "application_source_modified": False,
                "provider_call_authorized": False,
                "transition": str(transition),
                "transition_sha256": readiness.sha256_file(transition),
                "allocation_monitors": [
                    {"monitor": str(path)} for path in monitors
                ],
                "confirmation_gate": {"passed": True},
            }
            confirmation = {
                **payload, "result_id": bridge._fingerprint(payload),
            }
            confirmation_path = root / "confirmation.json"
            confirmation_path.write_text(
                json.dumps(confirmation) + "\n", encoding="utf-8",
            )
            with (
                mock.patch.object(
                    readiness.n8_confirmation, "validate_transition",
                    return_value=({}, {"compiler_graph": graph}),
                ),
                mock.patch.object(
                    readiness.n8_confirmation, "analyze_monitors",
                    return_value=confirmation,
                ) as replay,
            ):
                self.assertTrue(readiness.verified_collective_confirmation(
                    confirmation_path, graph_id,
                ))
                replay.assert_called_once_with(transition, monitors)

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

    def test_hidden_candidate_confirmation_requires_graph_expansion(self):
        producer_scout = {
            "schema_version": "gicc-producer-fission-oracle-analysis-v1",
            "correctness_gate": {"passed": True},
            "oracle_headroom_gate": {"passed": True},
        }
        producer = readiness.classify_producer(
            entry(), "promising", producer_scout, "confirmed", True,
        )
        self.assertEqual("graph_expansion_required", producer["status"])
        self.assertFalse(producer["provider_protocol_permitted"])
        self.assertFalse(producer["candidate_model_visible"])
        self.assertFalse(producer["current_suite_graph_expanded"])

        guarded_scout = {
            "schema_version": "gicc-guarded-early-trigger-analysis-v1",
            "correctness_gate": {"passed": True},
            "oracle_headroom_gate": {"passed": True},
        }
        guarded = readiness.classify_guarded_early_trigger(
            entry(), "promising", guarded_scout, "confirmed", True,
        )
        self.assertEqual("graph_expansion_required", guarded["status"])
        self.assertFalse(guarded["provider_protocol_permitted"])

    def test_hidden_confirmation_replays_before_graph_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transition = root / "transition.json"
            transition.write_text("{}\n", encoding="utf-8")
            monitors = [root / f"monitor{number}.json" for number in (1, 2, 3)]
            for path in monitors:
                path.write_text("{}\n", encoding="utf-8")
            payload = {
                "schema_version": "unit-hidden-confirmation-v1",
                "model_invoked": False,
                "application_source_modified": False,
                "provider_call_authorized": False,
                "transition": str(transition),
                "transition_sha256": readiness.sha256_file(transition),
                "allocation_monitors": [
                    {"monitor": str(path)} for path in monitors
                ],
                "correctness_gate": {"passed": True},
                "runtime_guard_gate": {"passed": True},
                "confirmation_gate": {"passed": True},
            }
            confirmation = {
                **payload, "result_id": bridge._fingerprint(payload),
            }
            path = root / "confirmation.json"
            path.write_text(json.dumps(confirmation) + "\n", encoding="utf-8")
            analyzer = mock.Mock()
            analyzer.analyze_monitors.return_value = confirmation
            self.assertTrue(readiness.verified_hidden_candidate_confirmation(
                path, schema=payload["schema_version"], analyzer=analyzer,
                label="unit-hidden", require_runtime_guard=True,
            ))
            analyzer.analyze_monitors.assert_called_once_with(
                transition, monitors,
            )

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

    def test_reused_loop_scout_stays_model_invisible(self):
        waiting = readiness.classify_reused_loop_descriptor(
            entry(), "waiting_predecessor", None,
        )
        self.assertEqual("awaiting_predecessor", waiting["status"])
        self.assertFalse(waiting["candidate_model_visible"])
        analysis = {
            "schema_version": "gicc-reused-loop-descriptor-analysis-v1",
            "correctness_gate": {"passed": True},
            "oracle_headroom_gate": {
                "passed": True,
                "paper_claim": False,
                "provider_protocol_permitted": False,
            },
        }
        result = readiness.classify_reused_loop_descriptor(
            entry(), "promising", analysis,
        )
        self.assertEqual("confirmation_required", result["status"])
        self.assertFalse(result["candidate_model_visible"])
        self.assertFalse(result["current_suite_graph_expanded"])
        self.assertFalse(result["provider_protocol_permitted"])

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
        with self.assertRaises(readiness.ReadinessError):
            readiness.classify_reused_loop_descriptor(
                entry(), "promising", {
                    "schema_version": (
                        "gicc-reused-loop-descriptor-analysis-v1"
                    ),
                    "correctness_gate": {"passed": True},
                    "oracle_headroom_gate": {
                        "passed": False,
                        "paper_claim": False,
                        "provider_protocol_permitted": False,
                    },
                },
            )


if __name__ == "__main__":
    unittest.main()
