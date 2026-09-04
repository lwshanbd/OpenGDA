import subprocess
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
FINALIZER = (
    PASS_ROOT / "experiments" /
    "continue_compiler_headroom_graph_refreezes.sh"
)


class CompilerHeadroomGraphRefreezeFinalizerTests(unittest.TestCase):
    def test_finalizer_is_scheduler_and_provider_free(self):
        subprocess.run(["bash", "-n", FINALIZER], check=True)
        text = FINALIZER.read_text(encoding="utf-8")
        self.assertNotIn("flux ", text)
        self.assertNotIn("anthropic", text.lower())
        self.assertNotIn("provider_call_authorized=true", text)
        self.assertNotIn("flux cancel", text)
        self.assertIn("flock -n 9", text)
        self.assertIn("verify_artifacts", text)

    def test_refreezes_are_strictly_serial(self):
        text = FINALIZER.read_text(encoding="utf-8")
        phases = [
            "waiting_producer_scout",
            "preparing_producer_expansion",
            "preparing_producer_refreeze",
            "waiting_guarded_confirmation",
            "preparing_guarded_expansion",
            "preparing_guarded_refreeze",
            "waiting_reused_confirmation",
            "preparing_reused_expansion",
            "preparing_reused_refreeze",
        ]
        offsets = [text.index(phase) for phase in phases]
        self.assertEqual(offsets, sorted(offsets))
        self.assertIn(
            'current_suite="$producer_refreeze_dir/suite.json"', text
        )
        self.assertIn(
            'current_suite="$guarded_refreeze_dir/suite.json"', text
        )
        self.assertIn(
            'current_suite="$reused_refreeze_dir/suite.json"', text
        )

    def test_each_positive_gate_has_separate_expand_and_refreeze(self):
        text = FINALIZER.read_text(encoding="utf-8")
        for family in ("producer", "guarded", "reused"):
            self.assertIn(f'if [[ ${family}_phase == confirmed ]]', text)
            self.assertIn(f'python3 "${family}_expander" prepare', text)
            self.assertIn(f'python3 "${family}_refreezer" prepare', text)
            self.assertIn(
                f'python3 "${family}_expander" verify-contained', text
            )
            self.assertIn(
                f'python3 "${family}_refreezer" verify-contained', text
            )
        self.assertIn('"jacobi=$jacobi_graph"', text)
        self.assertIn('"mm_minimal=$mm_graph"', text)
        self.assertIn(
            '"loop_lto=$portfolio/loop_lto/group-graph.json"', text
        )


if __name__ == "__main__":
    unittest.main()
