import json
import sys
import tempfile
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(PASS_ROOT / "experiments" / "collective"))

import compiler_collective_eval as controls
import gicc_collective_plan_bridge as collective
from test_gicc_collective_plan_bridge import PROFILE, inventory


class CompilerCollectiveEvalTests(unittest.TestCase):
    def setUp(self):
        self.graph = collective.make_graph(inventory(), PROFILE)

    def test_uniform_controls_cover_catalog_and_verify(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = controls.generate_controls(
                self.graph, root, "1" * 64, "2" * 64
            )
            controls.verify_manifest(self.graph, manifest, root)
            algorithms = {
                option["algorithm"]
                for option in self.graph["opportunities"][0][
                    "decision_slots"
                ][0]["options"]
            }
            self.assertEqual(algorithms, {
                arm["algorithm"] for arm in manifest["arms"]
            })
            self.assertEqual(
                self.graph["opportunities"][0]["joint_action_space_size"],
                manifest["joint_action_space_size"],
            )
            for arm in manifest["arms"]:
                hint = json.loads((root / arm["hint"]).read_text())
                selection = next(iter(hint["selections"].values()))
                self.assertEqual("uniform", selection["kind"])

    def test_analysis_constructs_a_compiler_bin_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = controls.generate_controls(
                self.graph, root, "1" * 64, "2" * 64
            )
            sizes = [1024, 8192, 1048576, 16777216]
            preferred = [
                "flat_double_tree", "flat_double_tree",
                "hierarchical_direct", "hierarchical_ring",
            ]
            logs = []
            for arm in manifest["arms"]:
                algorithm = arm["algorithm"]
                path = root / f"rep0-{arm['name']}.log"
                lines = [
                    f"COLLECTIVE_CONFIG plan={arm['name']} ranks=16 ppn=8 "
                    "runs=3 warmup=1"
                ]
                for index, size in enumerate(sizes):
                    latency = 10.0 if algorithm == preferred[index] else 20.0
                    lines.append(
                        f"RESULT plan={arm['name']} nodes=2 ranks=16 ppn=8 "
                        f"bytes={size} median_us={latency} errors=0"
                    )
                lines.append(
                    f"COLLECTIVE_DONE plan={arm['name']} total_errors=0"
                )
                path.write_text("\n".join(lines) + "\n")
                logs.append(path)
            summary = controls.analyze_logs(self.graph, manifest, logs)
            self.assertGreater(
                summary["aggregate"]["baseline_over_per_size_oracle"], 1.0
            )
            self.assertEqual(
                set(preferred),
                set(summary["aggregate"]["distinct_per_size_winners"]),
            )
            decision, hint = controls.oracle_decision(self.graph, summary)
            self.assertEqual(collective.DECISION_SCHEMA,
                             decision["schema_version"])
            self.assertFalse(hint["llm_metadata"]["model_invoked"])
            self.assertEqual(
                "size_policy", next(iter(hint["selections"].values()))["kind"]
            )


if __name__ == "__main__":
    unittest.main()
