import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    PASS_ROOT / "experiments" / "continue_compiler_headroom_recovery.sh"
)


class CompilerHeadroomRecoveryControllerTests(unittest.TestCase):
    def test_recovery_is_pdebug_only_non_cancelling_and_serial(self):
        text = SCRIPT.read_text()
        self.assertNotIn("flux cancel", text)
        self.assertNotIn("-q pci", text)
        self.assertIn("--filter=active", text)
        self.assertIn("wait_scheduler_idle producer_scout", text)
        self.assertIn("wait_scheduler_idle producer_confirmation", text)
        self.assertIn("wait_scheduler_idle guarded_scout", text)
        self.assertIn("wait_scheduler_idle n6_collective_scout", text)
        stages = [
            "running_producer_scout",
            "running_producer_confirmation",
            "running_guarded_scout",
            "running_guarded_successor",
            "running_n6_collective_scout",
        ]
        positions = [text.index(stage) for stage in stages]
        self.assertEqual(sorted(positions), positions)

    def test_recovery_has_no_model_or_provider_path(self):
        text = SCRIPT.read_text().lower()
        self.assertNotIn("claude", text)
        self.assertNotIn("anthropic", text)
        self.assertNotIn("model_trial", text)
        self.assertNotIn("provider", text)


if __name__ == "__main__":
    unittest.main()
