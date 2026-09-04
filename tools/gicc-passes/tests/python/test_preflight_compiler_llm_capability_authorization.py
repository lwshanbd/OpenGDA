import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = PASS_ROOT / "experiments"
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(EXPERIMENTS))
SCRIPT = EXPERIMENTS / "preflight_compiler_llm_capability_authorization.py"
SPEC = importlib.util.spec_from_file_location("capability_preflight", SCRIPT)
preflight = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = preflight
SPEC.loader.exec_module(preflight)


class CompilerLlmCapabilityAuthorizationPreflightTests(unittest.TestCase):
    def test_payload_records_no_inference_and_exact_serial_delivery(self):
        request = {
            "request_id": "sha256:" + "1" * 64,
            "provider_delivery": {
                "system_prompt_sha256": "2" * 64,
                "response_schema_sha256": "3" * 64,
                "views": [{
                    "view": view,
                    "prompt_sha256": str(index) * 64,
                    "independent_responses": 20,
                } for index, view in enumerate(
                    ("relational", "descriptors", "opaque"), start=4
                )],
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
        authorization_value = {
            "authorization_id": "sha256:" + "9" * 64,
        }
        authorization = {
            "provider": {
                "kind": "anthropic-claude-cli",
                "executable": "claude",
                "requested_model": "opus",
                "effort": "high",
                "cli_version": "unit-version",
                "fresh_session_per_trial": True,
                "structured_output": True,
            },
            "transport_retry": {
                "max_attempts_per_trial": 3,
                "timeout_seconds": 300,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path = root / "request.json"
            authorization_path = root / "authorization.json"
            request_path.write_text(json.dumps(request) + "\n")
            authorization_path.write_text(json.dumps(authorization_value) + "\n")
            result = preflight.payload(
                request, authorization_value, authorization, "unit-version",
                request_path, authorization_path,
            )
        self.assertEqual(
            "exact_authorization_verified_ready_for_serial_trials",
            result["status"],
        )
        self.assertEqual(60, result["execution_contract"][
            "total_conditional_calls"
        ])
        self.assertTrue(
            result["execution_contract"]["calls_strictly_sequential"]
        )
        self.assertFalse(result["boundary"]["provider_inference_invoked"])
        self.assertFalse(
            result["boundary"]["model_output_materialization_invoked"]
        )
        self.assertEqual([], result["boundary"]["model_tools"])

    def test_source_orders_exact_authorization_before_cli_inspection(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertLess(
            source.index("trials.verify_authorization"),
            source.index("trials.cli_version"),
        )
        self.assertLess(
            source.index("request_freezer.verify_bundle"),
            source.index("trials.verify_authorization"),
        )


if __name__ == "__main__":
    unittest.main()
