import importlib.util
import tempfile
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PASS_ROOT / "experiments" / "audit_compiler_llm_mainline_claims.py"
SPEC = importlib.util.spec_from_file_location("mainline_claims", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


class CompilerLlmMainlineClaimsTests(unittest.TestCase):
    def test_only_terminal_n6_chain_states_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state"
            state.write_text(
                "2026-09-04T00:00:00Z\tskipped\tscout_state=negative\n",
                encoding="utf-8",
            )
            self.assertEqual(
                ("skipped", "scout_state=negative"),
                audit.state_record(state),
            )
            state.write_text(
                "2026-09-04T00:00:00Z\twaiting_scout\tjob\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                audit.MainlineClaimError, "has not reached a terminal state"
            ):
                audit.state_record(state)

    def test_boundary_never_promotes_pre_inference_to_performance(self):
        self.assertFalse(audit.BOUNDARY["provider_call_authorized"])
        self.assertFalse(audit.BOUNDARY["provider_request_frozen"])
        self.assertFalse(audit.BOUNDARY["model_invoked"])
        self.assertFalse(
            audit.BOUNDARY["model_output_can_modify_application_source"]
        )


if __name__ == "__main__":
    unittest.main()
