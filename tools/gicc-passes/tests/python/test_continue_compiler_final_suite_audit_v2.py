import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PASS_ROOT / "experiments" / "continue_compiler_final_suite_audit_v2.sh"
PRIORITY = PASS_ROOT / "experiments" / "continue_compiler_priority_request_v2.sh"
TERMINAL = PASS_ROOT / "experiments" / "audit_compiler_terminal_negatives.py"


class ContinueCompilerFinalSuiteAuditV2Tests(unittest.TestCase):
    def test_finalizer_is_offline_and_uses_versioned_n6_evidence(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("flux ", text)
        self.assertNotIn("anthropic", text.lower())
        self.assertIn("compiler_collective_n6_confirmation_v2_20260904", text)
        self.assertIn("compiler_collective_capacity_n6_hierpipe_v2_20260904", text)
        self.assertIn("audit_compiler_terminal_negatives.py", text)
        self.assertIn("audit_compiler_llm_readiness_terminal.py", text)
        self.assertIn("audit_compiler_action_authority_terminal.py", text)

    def test_terminal_audit_precedes_optional_refreeze_and_claim_audits(self):
        text = SCRIPT.read_text(encoding="utf-8")
        stages = [
            "set_state auditing_terminal_negatives",
            "set_state replacing_collective_n6",
            'python3 "$input_auditor" emit',
            'python3 "$readiness_auditor" emit',
            'python3 "$authority_auditor" emit',
            'python3 "$null_auditor"',
            'python3 "$protocol_auditor" emit',
        ]
        positions = [text.index(stage) for stage in stages]
        self.assertEqual(sorted(positions), positions)
        self.assertNotIn("prepare_confirmed_producer", text)
        self.assertNotIn("prepare_confirmed_guarded", text)
        self.assertNotIn("prepare_confirmed_reused", text)

    def test_terminal_replay_has_no_scheduler_or_provider_path(self):
        text = TERMINAL.read_text(encoding="utf-8")
        self.assertNotIn("run_flux", text)
        self.assertNotIn("anthropic", text.lower())
        self.assertIn("runtime_values_used_as_positive_performance_evidence", text)

    def test_priority_successor_only_freezes_one_v2_request(self):
        text = PRIORITY.read_text(encoding="utf-8")
        self.assertNotIn("flux ", text)
        self.assertNotIn(
            'python3 "$script_dir/run_compiler_llm_capability_trials.py"', text
        )
        self.assertIn("compiler_final_suite_audit_v2_20260904", text)
        self.assertIn("compiler_collective_capacity_n6_hierpipe_v2_20260904", text)
        self.assertIn('python3 "$selector" freeze', text)
        self.assertIn('python3 "$selector" verify', text)


if __name__ == "__main__":
    unittest.main()
