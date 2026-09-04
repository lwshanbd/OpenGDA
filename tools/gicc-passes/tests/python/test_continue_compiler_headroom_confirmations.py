import subprocess
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    PASS_ROOT / "experiments" / "continue_compiler_headroom_confirmations.sh"
)


class CompilerHeadroomSuccessorTests(unittest.TestCase):
    def test_successor_is_syntax_checked_and_model_free(self):
        subprocess.run(["bash", "-n", SCRIPT], check=True)
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("-q pci", text)
        self.assertNotIn("anthropic", text.lower())
        self.assertNotIn("provider", text.lower())
        self.assertNotIn("model_trials", text)

    def test_jacobi_precedes_n8_and_both_use_serial_controllers(self):
        text = SCRIPT.read_text(encoding="utf-8")
        producer = text.index("continue_producer_fission_confirmation.sh")
        n8 = text.index("continue_compiler_collective_n8_confirmation.sh")
        self.assertLess(producer, n8)
        self.assertIn("waiting_producer_scout", text)
        self.assertIn("max_active_or_queued=1", (
            PASS_ROOT / "experiments" / "producer_fission"
            / "continue_producer_fission_confirmation.sh"
        ).read_text(encoding="utf-8"))
        self.assertIn("max_active_or_queued=1", (
            PASS_ROOT / "experiments" / "collective"
            / "continue_compiler_collective_n8_confirmation.sh"
        ).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
