import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PASS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PASS_ROOT / "experiments" / "collective"))

import run_compiler_collective_model_trials as trials
from test_gicc_collective_plan_bridge import PROFILE, inventory


class CompilerCollectiveModelTrialTests(unittest.TestCase):
    def setUp(self):
        self.request = {
            "request_id": "sha256:" + "1" * 64,
            "provider_delivery": {
                "system_prompt_sha256": "2" * 64,
                "response_schema_sha256": "3" * 64,
                "views": [{
                    "view": view,
                    "prompt_sha256": str(index) * 64,
                    "independent_responses": 20,
                } for index, view in enumerate(
                    ("relational", "descriptors", "opaque"), 4
                )],
            },
        }
        payload = {
            "schema_version": trials.AUTHORIZATION_SCHEMA,
            "granted": True,
            "request_id": self.request["request_id"],
            "system_prompt_sha256": "2" * 64,
            "response_schema_sha256": "3" * 64,
            "views": {
                row["view"]: {
                    "prompt_sha256": row["prompt_sha256"],
                    "independent_responses": 20,
                } for row in self.request["provider_delivery"]["views"]
            },
            "data_boundary": {
                "model_tools": [],
                "source_visible": False,
                "llvm_ir_visible": False,
                "evaluation_runtime_labels_visible": False,
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
        self.authorization = dict(payload)
        self.authorization["authorization_id"] = trials.bridge._fingerprint(
            payload
        )

    def test_authorization_binds_every_prompt_and_provider_setting(self):
        verified = trials.verify_authorization(
            self.authorization, self.request
        )
        self.assertEqual("opus", verified["provider"]["requested_model"])
        changed = copy.deepcopy(self.authorization)
        changed["views"]["opaque"]["prompt_sha256"] = "9" * 64
        payload = dict(changed)
        payload.pop("authorization_id")
        changed["authorization_id"] = trials.bridge._fingerprint(payload)
        with self.assertRaisesRegex(trials.TrialError, "exact Gate-E inputs"):
            trials.verify_authorization(changed, self.request)

    def test_ungranted_or_mutated_authorization_is_rejected(self):
        ungranted = copy.deepcopy(self.authorization)
        ungranted["granted"] = False
        payload = dict(ungranted)
        payload.pop("authorization_id")
        ungranted["authorization_id"] = trials.bridge._fingerprint(payload)
        with self.assertRaisesRegex(trials.TrialError, "not explicitly granted"):
            trials.verify_authorization(ungranted, self.request)
        mutated = copy.deepcopy(self.authorization)
        mutated["provider"]["requested_model"] = "different"
        with self.assertRaisesRegex(trials.TrialError, "authorization_id"):
            trials.verify_authorization(mutated, self.request)

    def test_provider_command_disables_tools_and_sessions(self):
        provider = self.authorization["provider"]
        command = trials.provider_command(
            provider, "system\n", {"type": "object"}
        )
        self.assertEqual("", command[command.index("--tools") + 1])
        self.assertIn("--safe-mode", command)
        self.assertIn("--no-session-persistence", command)
        self.assertEqual("opus", command[command.index("--model") + 1])

    def test_trial_order_is_three_views_times_twenty(self):
        order = trials.expected_trials(self.request)
        self.assertEqual(60, len(order))
        self.assertEqual(("relational", 1), order[0])
        self.assertEqual(("opaque", 20), order[-1])

    @mock.patch.object(trials.subprocess, "run")
    def test_malformed_successful_output_is_scored_once_and_archived(self, run):
        run.return_value = SimpleNamespace(
            returncode=0, stdout="not-json\n", stderr="",
        )
        graph = trials.plans.make_graph(inventory(), PROFILE)
        authorization = dict(self.authorization)
        authorization.pop("authorization_id")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            record = trials.run_one(
                view="relational", trial=1, prompt="compiler facts\n",
                graph=graph, request=self.request,
                authorization=authorization,
                authorization_id=self.authorization["authorization_id"],
                system_prompt="system\n", response_schema={"type": "object"},
                output_dir=output, observed_cli_version="test-version",
            )
            self.assertTrue(record["provider_call_succeeded"])
            self.assertFalse(record["bridge_accepted"])
            self.assertTrue(record["fallback_applied"])
            self.assertEqual(1, len(record["attempts"]))
            self.assertEqual(1, run.call_count)
            self.assertEqual(
                ("relational", 1),
                trials.verify_archived_run(
                    record, output, self.request,
                    self.authorization["authorization_id"],
                ),
            )
            response = output / "relational/trial01/response.json"
            response.write_text("{}\n")
            with self.assertRaisesRegex(trials.TrialError, "response changed"):
                trials.verify_archived_run(
                    record, output, self.request,
                    self.authorization["authorization_id"],
                )


if __name__ == "__main__":
    unittest.main()
