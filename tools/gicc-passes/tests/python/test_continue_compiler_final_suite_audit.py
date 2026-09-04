import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PASS_ROOT / "experiments" / "continue_compiler_final_suite_audit.sh"


class ContinueCompilerFinalSuiteAuditTests(unittest.TestCase):
    def test_successor_is_single_offline_finalizer(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("flux ", text)
        self.assertNotIn("anthropic", text.lower())
        self.assertIn("compiler_headroom_recovery_20260904.state", text)
        self.assertIn("compiler_collective_n6_confirmation_20260904.chain.state", text)
        self.assertIn("_20260904_r2", text)

    def test_transition_and_audit_order_is_serial(self):
        text = SCRIPT.read_text(encoding="utf-8")
        stages = [
            'set_state merging_producer',
            'set_state merging_guarded',
            'set_state merging_reused',
            'set_state replacing_collective_n6',
            'python3 "$input_auditor" emit',
            'python3 "$readiness_script" emit',
            'python3 "$authority_script"',
            'python3 "$null_auditor"',
            'python3 "$protocol_auditor" emit',
        ]
        positions = [text.index(stage) for stage in stages]
        self.assertEqual(sorted(positions), positions)
        self.assertIn('--graph-refreeze-finalizer "$successor"', text)


if __name__ == "__main__":
    unittest.main()
