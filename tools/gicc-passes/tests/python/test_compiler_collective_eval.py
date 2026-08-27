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

            decision, hint = controls.canary_decision(self.graph)
            self.assertEqual(collective.DECISION_SCHEMA,
                             decision["schema_version"])
            selection = next(iter(hint["selections"].values()))
            self.assertEqual("size_policy", selection["kind"])
            self.assertEqual(3, len({
                rule["target_id"] for rule in selection["rules"]
            }))
            self.assertFalse(hint["llm_metadata"]["model_invoked"])

            ir = root / "canary.ll"
            metadata = [
                "define void @run(i32 %count) {",
                "  %c0 = icmp ult i32 %count, 1025",
                "  %c1 = icmp ult i32 %count, 65537",
                "  %c2 = icmp ult i32 %count, 2097153",
            ]
            for index, rule in enumerate(selection["rules"]):
                metadata.append(
                    f'call void @candidate{index}(), '
                    f'!gicc.collective.candidate_id !{index * 2}, '
                    f'!gicc.collective.target_id !{index * 2 + 1}'
                )
                metadata.append(
                    f'!{index * 2} = !{{!"{selection["candidate_id"]}"}}'
                )
                metadata.append(
                    f'!{index * 2 + 1} = !{{!"{rule["target_id"]}"}}'
                )
            metadata.append("}")
            ir.write_text("\n".join(metadata) + "\n")
            controls.verify_plan_ir(self.graph, hint, ir)
            ir.write_text(ir.read_text().replace("65537", "65538"))
            with self.assertRaisesRegex(controls.EvalError, "policy cutoffs"):
                controls.verify_plan_ir(self.graph, hint, ir)

            audit = controls.audit_capacity(self.graph)
            self.assertEqual(4 ** 4, audit["enumerated_action_count"])
            self.assertEqual(4 ** 4, audit[
                "unique_composite_candidate_id_count"
            ])
            self.assertEqual(
                {"size_policy": 252, "uniform": 4},
                audit["materializer_kind_counts"],
            )
            self.assertFalse(audit["model_invoked"])

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
            self.assertAlmostEqual(2.0, summary["aggregate"][
                "baseline_over_compiler_bin_oracle"
            ])
            self.assertTrue(summary["gate_c"]["passed"])
            decision, hint = controls.oracle_decision(self.graph, summary)
            self.assertEqual(collective.DECISION_SCHEMA,
                             decision["schema_version"])
            self.assertFalse(hint["llm_metadata"]["model_invoked"])
            self.assertEqual(
                "size_policy", next(iter(hint["selections"].values()))["kind"]
            )

            prompt = root / "relational.txt"
            prompt.write_text(collective.render_prompt(self.graph))
            oracle_response = root / "oracle-response.json"
            oracle_response.write_text(json.dumps(decision))
            invalid_response = root / "invalid-response.json"
            invalid_response.write_text("not JSON\n")
            scores = controls.score_decisions(
                self.graph, summary, prompt, "relational",
                [oracle_response, invalid_response],
            )
            self.assertEqual(1, scores["aggregate"]["accepted_count"])
            self.assertEqual(0.5, scores["aggregate"]["invalid_output_rate"])
            self.assertTrue(scores["responses"][0][
                "exact_compiler_bin_oracle_policy"
            ])
            self.assertFalse(scores["responses"][1]["accepted"])
            self.assertTrue(scores["runtime_confirmation_required"])

    def test_gate_a_qualification_requires_exact_runtime_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = controls.generate_controls(
                self.graph, root, "1" * 64, "2" * 64
            )
            path = root / "smoke.log"
            path.write_text(
                "COLLECTIVE_CONFIG plan=baseline_auto ranks=16 ppn=8 "
                "runs=1 warmup=0\n"
                "RESULT plan=baseline_auto nodes=2 ranks=16 ppn=8 bytes=1024 "
                "median_us=10.0 errors=0\n"
                "RESULT plan=baseline_auto nodes=2 ranks=16 ppn=8 bytes=4096 "
                "median_us=11.0 errors=0\n"
                "COLLECTIVE_DONE plan=baseline_auto total_errors=0\n"
            )
            result = controls.qualify_logs(manifest, [path], "a")
            self.assertTrue(result["passed"])
            self.assertEqual("A", result["gate"])
            self.assertEqual(2, result["logs"]["baseline_auto"]["result_rows"])

            invalid = path.read_text().replace("nodes=2", "nodes=1", 1)
            path.write_text(invalid)
            with self.assertRaisesRegex(controls.EvalError, "topology"):
                controls.qualify_logs(manifest, [path], "a")

    def test_gate_c_fails_when_uniform_baseline_wins_every_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = controls.generate_controls(
                self.graph, root, "1" * 64, "2" * 64
            )
            logs = []
            for arm in manifest["arms"]:
                path = root / f"{arm['name']}.log"
                latency = 10.0 if arm["algorithm"] == "baseline_auto" else 20.0
                lines = [
                    f"COLLECTIVE_CONFIG plan={arm['name']} ranks=16 ppn=8 "
                    "runs=3 warmup=1"
                ]
                for size in [1024, 8192, 1048576, 16777216]:
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
            self.assertFalse(summary["gate_c"]["passed"])
            self.assertEqual(
                ["baseline_auto"],
                summary["aggregate"]["distinct_per_size_winners"],
            )


if __name__ == "__main__":
    unittest.main()
