import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PASS_ROOT / "experiments" / "continue_compiler_priority_request.sh"


class ContinueCompilerPriorityRequestTests(unittest.TestCase):
    def test_successor_is_offline_and_freezes_only(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("flux ", text)
        self.assertNotIn(
            'python3 "$script_dir/run_compiler_llm_capability_trials.py"',
            text,
        )
        self.assertIn("compiler_final_suite_audit_20260904", text)
        self.assertIn('python3 "$selector" freeze', text)
        self.assertIn('python3 "$selector" verify', text)

    def test_all_final_graph_families_are_resolved(self):
        text = SCRIPT.read_text(encoding="utf-8")
        for label in (
            "jacobi", "mm_minimal", "loop_lto", "minimod", "mixed_lto",
            "coalescing_placement", "collective_n6", "collective_n8",
        ):
            self.assertIn(label, text)


if __name__ == "__main__":
    unittest.main()
