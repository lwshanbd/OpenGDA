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

    def test_device_ir_requires_all_proxy_ring_reservations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "device.ll"
            kernel = controls.HDIR_DEVICE_KERNEL
            path.write_text(
                f"define protected amdgpu_kernel void @{kernel}() {{\n"
                "  %a = cmpxchg ptr null, i64 0, i64 1 monotonic monotonic\n"
                "  %b = cmpxchg ptr null, i64 0, i64 1 monotonic monotonic\n"
                "  %c = cmpxchg ptr null, i64 0, i64 1 monotonic monotonic\n"
                "  %d = cmpxchg ptr null, i64 0, i64 1 monotonic monotonic\n"
                "  ret void\n}\n"
            )
            self.assertEqual(4, controls.verify_device_ir(path))
            path.write_text(
                path.read_text().replace("  %d = cmpxchg", "  %d = add")
            )
            with self.assertRaisesRegex(controls.EvalError, "3/4"):
                controls.verify_device_ir(path)

    def test_build_provenance_freezes_same_build_dependency_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            build = repo / "build" / "arm"
            source = (repo / "tools/gicc-passes/experiments/collective" /
                      "compiler_collective_eval.cpp")
            catalog = source.with_name("compiler_collective_catalog.hpp")
            common = repo / "examples/proxy/coll_common.hpp"
            for path in (source, catalog, common):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"// {path.name}\n")
            build.mkdir(parents=True)

            def materialize(name, text="artifact\n"):
                path = build / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
                return path

            build_script = materialize("build.sh")
            evaluator = materialize("eval.py")
            plugin = materialize("passes.so")
            hint = materialize("hint.json", "{}\n")
            commands = materialize("commands.jsonl", "")
            controls.record_build_command(commands, [], reset=True)
            controls.record_build_command(
                commands, [sys.executable, "--version"], reset=False
            )
            depfile = materialize(
                "eval.d",
                f"{build / 'eval.o'}: {source} {catalog} \\\n {common}\n",
            )
            inputs = {
                "benchmark_source": source,
                "catalog_source": catalog,
                "build_script": build_script,
                "evaluator": evaluator,
                "compiler": Path(sys.executable),
                "pass_plugin": plugin,
                "collective_hint": hint,
            }
            artifact_names = {
                "eval_object": "eval.o",
                "inventory": "inventory.json",
                "host_ir": "materialized.ll",
                "device_ir": "materialized-device.ll",
                "device_ir_audit": "device-ir-audit.log",
                "compile_log": "compile.log",
                "host_ir_log": "ir.log",
                "device_ir_log": "device-ir.log",
                "command_log": "commands.jsonl",
                "binary": "compiler_collective_eval",
                "link_log": "link.log",
                "runtime_helpers_object": "obj/runtime_helpers.o",
                "proxy_thread_object": "obj/proxy_thread.o",
                "proxy_libfabric_object": "obj/proxy_libfabric.o",
            }
            artifacts = {
                role: commands if role == "command_log" else materialize(name)
                for role, name in artifact_names.items()
            }
            environment = {
                "GICC_MODE": "lower",
                "GICC_COLLECTIVE_ONLY": "1",
                "GICC_COLLECTIVE_HINT_IN": str(hint),
                "GICC_HINT_IN": None,
                "GICC_FEATURES_OUT": None,
            }
            manifest = controls.generate_build_provenance(
                mode="lower", repo_root=repo, build_root=build,
                inputs=inputs, artifacts=artifacts,
                dependency_files=[depfile], environment=environment,
                commands_path=commands,
            )
            manifest_path = build / "build-provenance.json"
            manifest_path.write_text(json.dumps(manifest))
            controls.verify_build_provenance(
                manifest, manifest_path=manifest_path, repo_root=repo
            )
            closure = manifest["dependency_closure"]
            self.assertEqual(3, closure["file_count"])
            runtime_records = {
                item["role"]: item for item in manifest["artifacts"]
                if item["role"].endswith("_object")
            }
            self.assertEqual(
                {"build"},
                {item["locator"]["scope"]
                 for item in runtime_records.values()},
            )

            artifacts["runtime_helpers_object"].write_text("tampered\n")
            with self.assertRaisesRegex(controls.EvalError, "hash mismatch"):
                controls.verify_build_provenance(
                    manifest, manifest_path=manifest_path, repo_root=repo
                )

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
