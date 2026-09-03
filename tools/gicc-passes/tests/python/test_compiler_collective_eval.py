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

    def test_v3_requires_a_complete_passed_pdebug_gate_a_monitor(self):
        contract = {
            "label": "baseline_auto_gate_a_scopefix",
            "nodes": 2,
            "ranks": 16,
            "ppn": 8,
            "runs": 1,
            "warmup": 0,
        }
        monitor = {
            "schema_version": "gicc-collective-job-monitor-v1",
            "state": "passed",
            "job_id": "test-job",
            "expected": {**contract, "sizes": [1024, 4096]},
            "benchmark": {
                "config": contract,
                "results": {"1024": 10.0, "4096": 11.0},
                "total_errors": 0,
            },
            "scheduler": {"exit_code": 0, "exceptions": []},
            "jobspec": {"queue": "pdebug"},
        }
        controls._validate_gate_a_monitor(monitor)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stdout = root / "gate-a.out"
            stderr = root / "gate-a.err"
            stdout.write_text(
                "COLLECTIVE_CONFIG plan=baseline_auto_gate_a_scopefix "
                "ranks=16 ppn=8 runs=1 warmup=0\n"
                "RESULT plan=baseline_auto_gate_a_scopefix nodes=2 ranks=16 "
                "ppn=8 bytes=1024 median_us=10.0 errors=0\n"
                "RESULT plan=baseline_auto_gate_a_scopefix nodes=2 ranks=16 "
                "ppn=8 bytes=4096 median_us=11.0 errors=0\n"
                "COLLECTIVE_DONE plan=baseline_auto_gate_a_scopefix "
                "total_errors=0\n"
            )
            stderr.write_text("")
            monitor["stdout_bytes"] = stdout.stat().st_size
            monitor["stderr_bytes"] = 0
            controls._validate_gate_a_logs(monitor, stdout, stderr)
            stdout.write_text(stdout.read_text().replace("11.0", "12.0"))
            with self.assertRaisesRegex(controls.EvalError, "timings"):
                controls._validate_gate_a_logs(monitor, stdout, stderr)
        monitor["jobspec"]["queue"] = "pci"
        with self.assertRaisesRegex(controls.EvalError, "clean pdebug"):
            controls._validate_gate_a_monitor(monitor)
        monitor["jobspec"]["queue"] = "pdebug"
        monitor["state"] = "monitoring"
        with self.assertRaisesRegex(controls.EvalError, "passed Gate-A"):
            controls._validate_gate_a_monitor(monitor)

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

    def test_confirmatory_analysis_pairs_same_allocation_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = controls.generate_controls(
                self.graph, root / "controls", "1" * 64, "2" * 64
            )
            arms = {arm["name"]: arm["algorithm"] for arm in manifest["arms"]}
            alternatives = sorted(
                algorithm for algorithm in arms.values()
                if algorithm != "baseline_auto"
            )
            slots = self.graph["opportunities"][0]["decision_slots"]
            preferred = {}
            for size in controls.GATE_B_SIZES:
                slot_index = next(
                    index for index, slot in enumerate(slots)
                    if (slot["message_bytes"]["min"] is None
                        or size >= slot["message_bytes"]["min"])
                    and (slot["message_bytes"]["max"] is None
                         or size <= slot["message_bytes"]["max"])
                )
                preferred[size] = alternatives[slot_index % 2]

            def write_log(path, label, algorithm, runs, warmup):
                lines = [
                    f"COLLECTIVE_CONFIG plan={label} ranks=16 ppn=8 "
                    f"runs={runs} warmup={warmup}"
                ]
                rows = {}
                for size in controls.GATE_B_SIZES:
                    latency = 10.0 if algorithm == preferred[size] else 20.0
                    rows[size] = latency
                    lines.append(
                        f"RESULT plan={label} nodes=2 ranks=16 ppn=8 "
                        f"bytes={size} median_us={latency} errors=0"
                    )
                lines.append(f"COLLECTIVE_DONE plan={label} total_errors=0")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("\n".join(lines) + "\n")
                return rows

            screen_logs = []
            for name, algorithm in arms.items():
                path = root / "screen" / f"{name}.out"
                write_log(path, name, algorithm, 3, 1)
                screen_logs.append(path)
            screen = controls.analyze_logs(self.graph, manifest, screen_logs)
            self.assertTrue(screen["gate_c"]["passed"])

            graph_path = root / "discovery/graph.json"
            graph_path.parent.mkdir(parents=True)
            graph_path.write_text(json.dumps(self.graph))
            screen_path = root / "gate-b/analysis.json"
            screen_path.parent.mkdir(parents=True)
            screen_path.write_text(json.dumps(screen))
            script_root = Path(controls.__file__).resolve().parent
            runner = script_root / "run_compiler_collective_replicate.sh"
            multi_monitor = (
                script_root / "monitor_compiler_collective_replicate.py"
            )
            artifact_paths = [
                root / "FROZEN_V3_MANIFEST.json",
                root / "controls/manifest.json",
                graph_path,
                screen_path,
                runner,
                multi_monitor,
            ]
            frozen_arms = []
            for name, algorithm in arms.items():
                binary = root / f"binaries/{name}/compiler_collective_eval"
                provenance = root / f"binaries/{name}/build-provenance.json"
                binary.parent.mkdir(parents=True)
                binary.write_text(f"binary for {algorithm}\n")
                provenance.write_text(json.dumps({"arm": name}))
                hint = root / f"controls/{name}-hint.json"
                artifact_paths.extend([binary, provenance, hint])
                frozen_arms.append({
                    "name": name,
                    "binary_sha256": controls._sha256(binary),
                    "hint_sha256": controls._sha256(hint),
                })
            freeze_payload = {
                "schema_version": controls.OFFLINE_FREEZE_SCHEMA,
                "bundle_version": 3,
                "graph": {"graph_id": self.graph["graph_id"]},
                "uniform_controls": {
                    "manifest_id": manifest["manifest_id"],
                    "arms": frozen_arms,
                },
            }
            freeze = dict(freeze_payload)
            freeze["manifest_id"] = controls.bridge._fingerprint(freeze_payload)
            artifact_paths[0].write_text(json.dumps(freeze))
            artifact_records = [
                {
                    "path": str(path.resolve()),
                    "sha256": controls._sha256(path),
                }
                for path in artifact_paths
            ]
            runner_sha256 = controls._sha256(runner)

            base_order = list(arms)
            monitor_paths = []
            for replicate in (1, 2, 3):
                shift = replicate - 1
                order = base_order[shift:] + base_order[:shift]
                benchmarks = {}
                for name, algorithm in arms.items():
                    path = root / f"rep{replicate}" / f"{name}.out"
                    rows = write_log(path, name, algorithm, 7, 2)
                    benchmarks[name] = {
                        "benchmark": {
                            "config": {
                                "label": name, "nodes": 2, "ranks": 16,
                                "ppn": 8, "runs": 7, "warmup": 2,
                            },
                            "results": {
                                str(size): rows[size]
                                for size in controls.GATE_B_SIZES
                            },
                            "total_errors": 0,
                        },
                        "stdout": {
                            "path": str(path),
                            "bytes": path.stat().st_size,
                            "sha256": controls._sha256(path),
                        },
                        "stderr": {},
                    }
                driver = root / f"rep{replicate}" / "driver.out"
                driver.write_text(
                    f"REPLICATE_CONFIG replicate={replicate} "
                    f"arms={' '.join(order)}\n"
                    + "".join(
                        f"REPLICATE_ARM_START replicate={replicate} arm={name}\n"
                        f"REPLICATE_ARM_DONE replicate={replicate} arm={name}\n"
                        for name in order
                    )
                    + f"REPLICATE_DONE replicate={replicate}\n"
                )
                monitor = {
                    "schema_version": (
                        "gicc-collective-replicate-job-monitor-v1"
                    ),
                    "state": "passed",
                    "replicate": replicate,
                    "job_id": f"job-{replicate}",
                    "expected": {
                        "nodes": 2, "ranks": 16, "ppn": 8,
                        "runs": 7, "warmup": 2,
                        "sizes": controls.GATE_B_SIZES,
                    },
                    "scheduler": {"exit_code": 0, "exception_types": []},
                    "jobspec": {
                        "queue": "pdebug",
                        "duration_seconds": 2700.0,
                        "embedded_script_sha256": runner_sha256,
                        "resources": [{
                            "type": "node",
                            "count": 2,
                            "with": [{
                                "type": "slot",
                                "count": 8,
                                "with": [
                                    {"type": "core", "count": 8},
                                    {"type": "gpu", "count": 1},
                                ],
                                "label": "task",
                            }],
                        }],
                        "command": [
                            "flux", "broker", "-c{{tmpdir}}/conf.json",
                            "{{tmpdir}}/script", str(root.resolve()),
                            str((root / f"rep{replicate}").resolve()),
                            str(replicate), *order,
                        ],
                    },
                    "resource_set": {
                        "nodelist": [f"tioga[{replicate}-{replicate + 1}]"],
                    },
                    "driver_stdout": {
                        "path": str(driver),
                        "bytes": driver.stat().st_size,
                        "sha256": controls._sha256(driver),
                    },
                    "benchmarks": benchmarks,
                    "artifacts": artifact_records,
                }
                monitor_path = root / f"rep{replicate}.monitor.json"
                monitor_path.write_text(json.dumps(monitor))
                monitor_paths.append(monitor_path)

            result = controls.confirmatory_analysis(
                self.graph, manifest, screen, monitor_paths
            )
            primary = result["primary_headroom_confirmation"]
            self.assertAlmostEqual(2.0, primary["point_estimate"])
            self.assertTrue(primary["confidence_interval_excludes_one"])
            self.assertEqual(
                27,
                result["paired_baseline_speedups"][
                    "screen_frozen_compiler_bin_oracle"
                ]["aggregate"]["bootstrap_samples"],
            )

            binary = root / f"binaries/{base_order[0]}/compiler_collective_eval"
            binary_bytes = binary.read_bytes()
            binary.write_bytes(binary_bytes + b"tampered\n")
            with self.assertRaisesRegex(controls.EvalError, "artifact changed"):
                controls.confirmatory_analysis(
                    self.graph, manifest, screen, monitor_paths
                )
            binary.write_bytes(binary_bytes)

            invalid = json.loads(monitor_paths[1].read_text())
            invalid_driver = Path(invalid["driver_stdout"]["path"])
            invalid_driver.write_text(
                "REPLICATE_CONFIG replicate=2 "
                f"arms={' '.join(base_order)}\n"
                + "".join(
                    f"REPLICATE_ARM_START replicate=2 arm={name}\n"
                    f"REPLICATE_ARM_DONE replicate=2 arm={name}\n"
                    for name in base_order
                )
                + "REPLICATE_DONE replicate=2\n"
            )
            invalid["driver_stdout"].update({
                "bytes": invalid_driver.stat().st_size,
                "sha256": controls._sha256(invalid_driver),
            })
            invalid["jobspec"]["command"][-len(base_order):] = base_order
            monitor_paths[1].write_text(json.dumps(invalid))
            with self.assertRaisesRegex(controls.EvalError, "not rotated"):
                controls.confirmatory_analysis(
                    self.graph, manifest, screen, monitor_paths
                )

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
