import copy
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PROXY = ROOT / "examples/proxy"
sys.path.insert(0, str(PROXY))

import gicc_llm_bridge as bridge
import run_compiler_lto_llm_calibrated_trials as trials


def request():
    return {
        "request_id": "sha256:" + "1" * 64,
        "provider_delivery": {
            "independent_responses": 20,
            "files": [
                {"role": "system_prompt", "sha256": "a" * 64},
                {"role": "user_prompt", "sha256": "b" * 64},
                {"role": "response_schema", "sha256": "c" * 64},
            ],
        },
    }


def authorization(req, granted=True):
    payload = {
        "schema_version": trials.AUTHORIZATION_SCHEMA,
        "granted": granted,
        "request_id": req["request_id"],
        "system_prompt_sha256": "a" * 64,
        "prompt_sha256": "b" * 64,
        "response_schema_sha256": "c" * 64,
        "independent_responses": 20,
        "data_boundary": {
            "source_visible": False,
            "llvm_ir_visible": False,
            "evaluation_runtime_labels_visible": False,
            "evaluation_oracle_visible": False,
            "calibration_labels_visible": True,
            "model_tools": [],
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
    value = dict(payload)
    value["authorization_id"] = bridge._fingerprint(payload)
    return value


class RunCompilerLtoLlmCalibratedTrialsTests(unittest.TestCase):
    def test_exact_authorization_is_accepted(self):
        req = request()
        value = authorization(req)
        payload = trials.verify_authorization(value, req)
        self.assertTrue(payload["granted"])

    def test_missing_or_stale_authorization_is_rejected(self):
        req = request()
        denied = authorization(req, granted=False)
        with self.assertRaisesRegex(trials.TrialError, "not explicitly"):
            trials.verify_authorization(denied, req)

        stale = authorization(req)
        stale["prompt_sha256"] = "d" * 64
        payload = dict(stale)
        payload.pop("authorization_id")
        stale["authorization_id"] = bridge._fingerprint(payload)
        with self.assertRaisesRegex(trials.TrialError, "exact request"):
            trials.verify_authorization(stale, req)

        tampered = copy.deepcopy(authorization(req))
        tampered["provider"]["requested_model"] = "different"
        with self.assertRaisesRegex(trials.TrialError, "authorization_id"):
            trials.verify_authorization(tampered, req)

        extra = authorization(req)
        extra["unreviewed_option"] = True
        payload = dict(extra)
        payload.pop("authorization_id")
        extra["authorization_id"] = bridge._fingerprint(payload)
        with self.assertRaisesRegex(trials.TrialError, "frozen schema"):
            trials.verify_authorization(extra, req)

    def test_provider_command_disables_tools_and_sessions(self):
        provider = authorization(request())["provider"]
        command = trials.provider_command(provider, "system", {"type": "object"})
        self.assertEqual("", command[command.index("--tools") + 1])
        self.assertIn("--safe-mode", command)
        self.assertIn("--no-session-persistence", command)

    def test_authorization_archive_is_idempotent_and_immutable(self):
        value = authorization(request())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = trials.archive_authorization(value, output)
            self.assertEqual(value, trials.read_json(path))
            self.assertEqual(path, trials.archive_authorization(value, output))
            changed = copy.deepcopy(value)
            changed["authorization_id"] = "sha256:" + "0" * 64
            with self.assertRaisesRegex(trials.TrialError, "differs"):
                trials.archive_authorization(changed, output)


if __name__ == "__main__":
    unittest.main()
