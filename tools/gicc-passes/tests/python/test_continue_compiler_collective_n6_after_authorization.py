import subprocess
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
COLLECTIVE = PASS_ROOT / "experiments" / "collective"
SUCCESSOR = (
    COLLECTIVE / "continue_compiler_collective_n6_after_authorization.sh"
)
RUNTIME_ANALYZER = (
    COLLECTIVE / "analyze_collective_n6_llm_runtime_validation.py"
)


class ContinueCollectiveN6AfterAuthorizationTests(unittest.TestCase):
    def test_successor_is_guarded_and_delegates_serial_scheduler_work(self):
        subprocess.run(["bash", "-n", SUCCESSOR], check=True)
        text = SUCCESSOR.read_text(encoding="utf-8")
        self.assertNotIn("-q pci", text)
        self.assertNotIn("flux batch", text)
        self.assertNotIn("flux cancel", text)
        self.assertIn('entry.get("label") != "collective_n6"', text)
        self.assertIn("priority selection and request ID differ", text)
        self.assertIn('python3 "$trial_runner"', text)
        self.assertIn('"$screen_controller"', text)
        self.assertIn('"$runtime_builder"', text)
        self.assertIn('"$runtime_controller"', text)
        self.assertIn('python3 "$paper_auditor"', text)
        self.assertLess(
            text.index('entry.get("label") != "collective_n6"'),
            text.index('python3 "$trial_runner"'),
        )
        self.assertLess(
            text.index('python3 "$trial_runner"'),
            text.index('"$screen_controller" "$n6_bundle"'),
        )
        self.assertLess(
            text.index('"$screen_controller" "$n6_bundle"'),
            text.index('"$runtime_controller" "$plan_dir"'),
        )
        self.assertLess(
            text.index('"$runtime_controller" "$plan_dir"'),
            text.index('python3 "$paper_auditor"'),
        )

    def test_runtime_analysis_has_a_read_only_reverification_mode(self):
        text = RUNTIME_ANALYZER.read_text(encoding="utf-8")
        self.assertIn('"--verify", action="store_true"', text)
        self.assertIn("runtime analysis does not match current evidence", text)


if __name__ == "__main__":
    unittest.main()
