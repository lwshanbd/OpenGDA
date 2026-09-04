import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PASS_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PASS_ROOT / "experiments"))
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(TEST_ROOT))

import gicc_comm_plan_bridge as structural
import gicc_llm_bridge as llm
import run_compiler_llm_capability_trials as trials
from test_gicc_comm_plan_bridge import PLATFORM, coalescable_feature


def request_and_authorization():
    views = [{
        "view": view,
        "prompt_sha256": str(index) * 64,
        "independent_responses": 20,
    } for index, view in enumerate(
        ("relational", "descriptors", "opaque"), 4
    )]
    request = {
        "request_id": "sha256:" + "1" * 64,
        "provider_delivery": {
            "system_prompt_sha256": "2" * 64,
            "response_schema_sha256": "3" * 64,
            "views": views,
            "trial_order": {
                "kind": "response_index_major_rotating_views",
                "base_view_order": [
                    "relational", "descriptors", "opaque",
                ],
                "rotation_offset_for_trial": (
                    "(trial - 1) modulo view count"
                ),
            },
            "total_conditional_calls": 60,
        },
    }
    payload = {
        "schema_version": trials.AUTHORIZATION_SCHEMA,
        "granted": True,
        "request_id": request["request_id"],
        "provider_delivery": trials.delivery_contract(request),
        "data_boundary": {
            "model_tools": [],
            "source_visible": False,
            "source_locations_visible": False,
            "llvm_ir_visible": False,
            "evaluation_runtime_labels_visible": False,
            "evaluation_oracle_visible": False,
            "model_may_generate_code_or_ir": False,
        },
        "provider": {
            "kind": "anthropic-claude-cli",
            "executable": "claude",
            "cli_version": "test-version",
            "requested_model": "opus",
            "effort": "high",
            "fresh_session_per_trial": True,
            "structured_output": True,
        },
        "transport_retry": {
            "max_attempts_per_trial": 3,
            "timeout_seconds": 300,
        },
    }
    authorization = dict(payload)
    authorization["authorization_id"] = llm._fingerprint(payload)
    return request, authorization


class CompilerLlmCapabilityTrialTests(unittest.TestCase):
    def setUp(self):
        self.request, self.authorization = request_and_authorization()

    def test_authorization_binds_delivery_provider_and_boundary(self):
        verified = trials.verify_authorization(
            self.authorization, self.request
        )
        self.assertEqual("opus", verified["provider"]["requested_model"])
        changed = copy.deepcopy(self.authorization)
        changed["provider_delivery"]["views"]["opaque"][
            "prompt_sha256"
        ] = "9" * 64
        payload = dict(changed)
        payload.pop("authorization_id")
        changed["authorization_id"] = llm._fingerprint(payload)
        with self.assertRaisesRegex(
            trials.CapabilityTrialError, "exact provider delivery"
        ):
            trials.verify_authorization(changed, self.request)

        changed = copy.deepcopy(self.authorization)
        changed["data_boundary"]["source_visible"] = True
        payload = dict(changed)
        payload.pop("authorization_id")
        changed["authorization_id"] = llm._fingerprint(payload)
        with self.assertRaisesRegex(
            trials.CapabilityTrialError, "compiler-only data boundary"
        ):
            trials.verify_authorization(changed, self.request)

    def test_provider_command_disables_tools_and_context_persistence(self):
        command = trials.provider_command(
            self.authorization["provider"], "system\n", {"type": "object"}
        )
        self.assertEqual("", command[command.index("--tools") + 1])
        self.assertIn("--safe-mode", command)
        self.assertIn("--no-session-persistence", command)

    def test_trial_order_rotates_views_by_response_index(self):
        order = trials.expected_trials(self.request)
        self.assertEqual(60, len(order))
        self.assertEqual([
            ("relational", 1), ("descriptors", 1), ("opaque", 1),
            ("descriptors", 2), ("opaque", 2), ("relational", 2),
            ("opaque", 3), ("relational", 3), ("descriptors", 3),
        ], order[:9])

    def test_offline_archive_verifier_rejects_incomplete_index(self):
        graph = structural.make_opportunity_graph(llm.make_dossier(
            [coalescable_feature()], PLATFORM,
        ))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            trials.write_json_atomic(
                output / "authorization.json", self.authorization,
            )
            trials.write_json_atomic(output / "run-index.json", {
                "schema_version": trials.INDEX_SCHEMA,
                "status": "running",
                "request_id": self.request["request_id"],
                "authorization_id": self.authorization["authorization_id"],
                "runs": [],
            })
            with self.assertRaisesRegex(
                trials.CapabilityTrialError, "archive is not complete"
            ):
                trials.verify_complete_archive(
                    request=self.request,
                    authorization_value=self.authorization,
                    graph=graph,
                    output_dir=output,
                )

    @mock.patch.object(trials.subprocess, "run")
    def test_semantic_failure_is_archived_once_with_compiler_fallback(self, run):
        run.return_value = SimpleNamespace(
            returncode=0, stdout="not-json\n", stderr="",
        )
        graph = structural.make_opportunity_graph(llm.make_dossier(
            [coalescable_feature()], PLATFORM,
        ))
        schema = structural.decision_response_schema(graph)
        prompt = "source-free compiler facts\n"
        system = "compiler selector\n"
        self.request["provider_delivery"]["views"][0][
            "prompt_sha256"
        ] = trials.sha256_bytes(prompt.encode())
        self.request["provider_delivery"]["system_prompt_sha256"] = (
            trials.sha256_bytes(system.encode())
        )
        self.request["provider_delivery"]["response_schema_sha256"] = (
            trials.sha256_bytes(
                (json.dumps(schema, indent=2, sort_keys=True) + "\n").encode()
            )
        )
        authorization = trials.authorization_payload(self.authorization)
        # run_one receives an already verified authorization payload; update
        # only the request-side hashes needed by archive verification.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            record = trials.run_one(
                view="relational", trial=1, prompt=prompt, graph=graph,
                request=self.request, authorization=authorization,
                authorization_id=self.authorization["authorization_id"],
                system_prompt=system, response_schema=schema,
                output_dir=output, observed_cli_version="test-version",
            )
            self.assertEqual(1, run.call_count)
            self.assertTrue(record["provider_call_succeeded"])
            self.assertIsNotNone(record["response_parse_error"])
            self.assertEqual(0, record["semantic_retry_count"])
            self.assertFalse(record["bridge_accepted"])
            self.assertTrue(record["fallback_applied"])
            self.assertTrue(
                record["compiler_hint_is_private_and_never_provider_input"]
            )
            self.assertEqual(1, len(record["attempts"]))

            archive_authorization = copy.deepcopy(authorization)
            archive_authorization["provider"]["cli_version"] = "test-version"
            self.assertEqual(
                ("relational", 1),
                trials.verify_archived_run(
                    record, output, self.request, archive_authorization,
                    self.authorization["authorization_id"], graph,
                ),
            )
            hint = output / "relational/trial01/private/compiler-hint.json"
            hint.write_text("{}\n")
            with self.assertRaisesRegex(
                trials.CapabilityTrialError, "response or hint changed"
            ):
                trials.verify_archived_run(
                    record, output, self.request, archive_authorization,
                    self.authorization["authorization_id"], graph,
                )


if __name__ == "__main__":
    unittest.main()
