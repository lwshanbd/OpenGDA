import json
import subprocess
import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
COLLECTIVE = PASS_ROOT / "experiments" / "collective"
ADAPTER = COLLECTIVE / "prepare_collective_n6_llm_policy_screen.py"
RUNNER = COLLECTIVE / "run_compiler_collective_n6_llm_controls.sh"
CONTROLLER = COLLECTIVE / "continue_compiler_collective_n6_llm_controls.sh"


class CollectiveN6LlmPolicyScreenTests(unittest.TestCase):
    def test_adapter_binds_n6_confirmation_and_resources(self):
        code = (
            "import json,sys; sys.path.insert(0,sys.argv[1]); "
            "import prepare_collective_n6_llm_policy_screen as x; "
            "print(json.dumps({"
            "'schema':x.base.CONFIRMATION_SCHEMA,"
            "'topology':x.base.TOPOLOGY_LABEL,"
            "'nodes':x.base.CONTROL_NODES,"
            "'ranks':x.base.CONTROL_RANKS,"
            "'runner':x.base.RUNNER_NAME,"
            "'controller':x.base.CONTROLLER_NAME}))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code, str(COLLECTIVE)],
            check=True, capture_output=True, text=True,
        )
        value = json.loads(completed.stdout)
        self.assertEqual("gicc-collective-n6-confirmation-v1", value["schema"])
        self.assertEqual("N6", value["topology"])
        self.assertEqual((6, 48), (value["nodes"], value["ranks"]))
        self.assertEqual(RUNNER.name, value["runner"])
        self.assertEqual(CONTROLLER.name, value["controller"])

    def test_runtime_controls_are_one_pdebug_job(self):
        for script in (RUNNER, CONTROLLER):
            subprocess.run(["bash", "-n", script], check=True)
            self.assertNotIn("-q pci", script.read_text(encoding="utf-8"))
        controller = CONTROLLER.read_text(encoding="utf-8")
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertEqual(1, controller.count("flux batch"))
        self.assertIn("flux batch -q pdebug -N6 -n48", controller)
        self.assertIn("flux run -N6 -n48", runner)
        self.assertEqual(3, runner.count("run_block "))
        self.assertNotIn("flux cancel", controller)


if __name__ == "__main__":
    unittest.main()
