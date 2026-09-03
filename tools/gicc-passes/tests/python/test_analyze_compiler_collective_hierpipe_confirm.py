import json
import sys
import tempfile
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PASS_ROOT / "experiments" / "collective"))

import analyze_compiler_collective_hierpipe_confirm as confirm


def scout_fixture():
    per_size = {
        str(size): {
            "winner": (
                "hierarchical_double_tree_pipe4"
                if size == 1024 else "hierarchical_double_tree"
            ),
        }
        for size in confirm.SIZES
    }
    payload = {
        "schema_version": "gicc-collective-hierpipe-scout-analysis-v1",
        "model_invoked": False,
        "application_source_modified": False,
        "hierarchical_pipeline_gate": {"passed": True},
        "aggregate": {
            "best_uniform_algorithm": "hierarchical_double_tree",
        },
        "per_size": per_size,
    }
    return {**payload, "result_id": confirm.fingerprint(payload)}


def uniform_rows():
    return {
        replicate: {
            "hierarchical_double_tree": {
                str(size): 300.0 for size in confirm.SIZES
            },
            "hierarchical_double_tree_pipe4": {
                str(size): 700.0 for size in confirm.SIZES
            },
            "hierarchical_double_tree_pipe8": {
                str(size): 1200.0 for size in confirm.SIZES
            },
        }
        for replicate in (1, 2, 3)
    }


class CompilerCollectiveHierpipeConfirmTests(unittest.TestCase):
    def test_stable_scout_frozen_crossover_passes(self):
        rows = uniform_rows()
        for replicate in rows:
            rows[replicate]["hierarchical_double_tree_pipe4"]["1024"] = 100.0
        result = confirm.analyze_rows(rows, scout_fixture())
        self.assertTrue(result["confirmation_gate"]["passed"])
        self.assertEqual(
            ["hierarchical_double_tree", "hierarchical_double_tree_pipe4"],
            result["aggregate"]["distinct_size_winners"],
        )
        primary = result["scout_frozen_policy"]["paired_speedup"]
        self.assertGreater(primary["lower_2_5_percent"], 1.0)
        self.assertEqual(27, primary["bootstrap_samples"])

    def test_first_position_outlier_does_not_confirm(self):
        rows = uniform_rows()
        rows[1]["hierarchical_double_tree"]["1024"] = 3000.0
        result = confirm.analyze_rows(rows, scout_fixture())
        self.assertFalse(result["confirmation_gate"]["passed"])
        persistence = result["scout_frozen_policy"][
            "nonuniform_winner_persistence"
        ]["1024"]
        self.assertEqual(1, persistence["wins_over_scout_uniform"])
        self.assertFalse(persistence["passed"])

    def test_scout_must_be_content_addressed_and_blocks_complete(self):
        scout = scout_fixture()
        scout["aggregate"]["best_uniform_algorithm"] = (
            "hierarchical_double_tree_pipe8"
        )
        with self.assertRaisesRegex(confirm.ConfirmError, "content-addressed"):
            confirm.analyze_rows(uniform_rows(), scout)
        with self.assertRaisesRegex(confirm.ConfirmError, "blocks 1, 2, and 3"):
            confirm.analyze_rows({1: uniform_rows()[1]}, scout_fixture())

    def test_complete_archive_reparses_raw_logs_and_rejects_forgery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            scout_dir = root / "scout"
            output = root / "output"
            (bundle / "discovery").mkdir(parents=True)
            (bundle / "inputs").mkdir()
            (bundle / "binaries").mkdir()
            (bundle / "controls").mkdir()
            scout_dir.mkdir()
            output.mkdir()
            for relative in (
                "FROZEN_V3_MANIFEST.json",
                "discovery/graph.json",
                "inputs/platform.json",
            ):
                path = bundle / relative
                path.write_text("{}\n")
            for arm in confirm.ARMS:
                binary_dir = bundle / "binaries" / arm
                binary_dir.mkdir()
                (binary_dir / "compiler_collective_eval").write_text(arm + "\n")
                (binary_dir / "build-provenance.json").write_text("{}\n")
                (bundle / "controls" / f"{arm}-hint.json").write_text("{}\n")

            scout_monitor = scout_dir / "monitor.json"
            scout_monitor.write_text("{}\n")
            scout = scout_fixture()
            scout_payload = dict(scout)
            scout_payload.pop("result_id")
            scout_payload.update({
                "monitor": str(scout_monitor.resolve()),
                "monitor_sha256": confirm.sha256(scout_monitor),
            })
            scout = {
                **scout_payload,
                "result_id": confirm.fingerprint(scout_payload),
            }
            scout_path = scout_dir / "analysis.json"
            scout_path.write_text(json.dumps(scout))

            script_dir = Path(confirm.__file__).resolve().parent
            artifacts = [
                bundle / "FROZEN_V3_MANIFEST.json",
                bundle / "discovery/graph.json",
                bundle / "inputs/platform.json",
                scout_path,
                scout_monitor,
                script_dir / "PROTOCOL_DRAFT.md",
                script_dir / "run_compiler_collective_hierpipe_confirm.sh",
                script_dir / "continue_compiler_collective_hierpipe_confirm.sh",
                script_dir / "monitor_compiler_collective_replicate.py",
                Path(confirm.__file__),
            ]
            for arm in confirm.ARMS:
                artifacts.extend([
                    bundle / f"binaries/{arm}/compiler_collective_eval",
                    bundle / f"binaries/{arm}/build-provenance.json",
                    bundle / f"controls/{arm}-hint.json",
                ])
            artifact_records = [
                {"path": str(path.resolve()), "sha256": confirm.sha256(path)}
                for path in artifacts
            ]
            resources = [{
                "type": "node", "count": 4,
                "with": [{
                    "type": "slot", "count": 8, "label": "task",
                    "with": [
                        {"type": "core", "count": 8},
                        {"type": "gpu", "count": 1},
                    ],
                }],
            }]
            source_rows = uniform_rows()
            for replicate in source_rows:
                source_rows[replicate][
                    "hierarchical_double_tree_pipe4"
                ]["1024"] = 100.0
            monitor_paths = []
            monitors = []
            for replicate in (1, 2, 3):
                rep_dir = output / f"rep{replicate}"
                rep_dir.mkdir()
                driver = rep_dir / "driver.out"
                driver_err = rep_dir / "driver.err"
                lines = [
                    f"HIERPIPE_CONFIRM_CONFIG replicate={replicate} "
                    f"arms={' '.join(confirm.ORDERS[replicate])}",
                ]
                for arm in confirm.ORDERS[replicate]:
                    lines.extend([
                        f"HIERPIPE_CONFIRM_ARM_START replicate={replicate} "
                        f"arm={arm}",
                        f"HIERPIPE_CONFIRM_ARM_DONE replicate={replicate} "
                        f"arm={arm}",
                    ])
                lines.append(f"HIERPIPE_CONFIRM_DONE replicate={replicate}")
                driver.write_text("\n".join(lines) + "\n")
                driver_err.write_text("")
                benchmarks = {}
                for arm in confirm.ARMS:
                    log = rep_dir / f"{arm}.out"
                    err = rep_dir / f"{arm}.err"
                    output_lines = [
                        f"COLLECTIVE_CONFIG plan={arm} ranks=32 ppn=8 "
                        "runs=7 warmup=2",
                    ]
                    for size in confirm.SIZES:
                        latency = source_rows[replicate][arm][str(size)]
                        output_lines.append(
                            f"RESULT plan={arm} nodes=4 ranks=32 ppn=8 "
                            f"bytes={size} median_us={latency:.3f} errors=0"
                        )
                    output_lines.append(
                        f"COLLECTIVE_DONE plan={arm} total_errors=0"
                    )
                    log.write_text("\n".join(output_lines) + "\n")
                    err.write_text("")
                    benchmark = confirm.monitor_base.validate_output(
                        log, arm, list(confirm.SIZES), 4, 32, 8, 7, 2,
                    )
                    benchmarks[arm] = {
                        "benchmark": benchmark,
                        "stdout": {
                            "path": str(log), "bytes": log.stat().st_size,
                            "sha256": confirm.sha256(log),
                        },
                        "stderr": {
                            "path": str(err), "bytes": 0,
                            "sha256": confirm.sha256(err),
                        },
                    }
                monitor = {
                    "schema_version": (
                        "gicc-collective-replicate-job-monitor-v1"
                    ),
                    "state": "passed",
                    "job_id": "one-job",
                    "replicate": replicate,
                    "expected": {
                        "nodes": 4, "ranks": 32, "ppn": 8,
                        "runs": 7, "warmup": 2,
                        "sizes": list(confirm.SIZES),
                    },
                    "scheduler": {"exit_code": 0, "exception_types": []},
                    "jobspec": {
                        "queue": "pdebug", "duration_seconds": 1800.0,
                        "resources": resources,
                        "embedded_script_sha256": confirm.sha256(
                            script_dir
                            / "run_compiler_collective_hierpipe_confirm.sh"
                        ),
                        "command": [
                            "flux", "broker", "{{tmpdir}}/script",
                            str(bundle), str(output),
                        ],
                    },
                    "resource_set": {"nodelist": ["tioga[1-4]"]},
                    "artifacts": artifact_records,
                    "driver_stdout": {
                        "path": str(driver), "bytes": driver.stat().st_size,
                        "sha256": confirm.sha256(driver),
                    },
                    "driver_stderr": {
                        "path": str(driver_err), "bytes": 0,
                        "sha256": confirm.sha256(driver_err),
                    },
                    "benchmarks": benchmarks,
                }
                monitor_path = rep_dir / "monitor.json"
                monitor_path.write_text(json.dumps(monitor))
                monitor_paths.append(monitor_path)
                monitors.append(monitor)
            result = confirm.analyze_monitors(scout_path, monitor_paths)
            self.assertTrue(result["confirmation_gate"]["passed"])
            self.assertEqual("one-job", result["job_id"])

            monitors[0]["benchmarks"]["hierarchical_double_tree"][
                "benchmark"
            ]["results"]["1024"] = 1.0
            monitor_paths[0].write_text(json.dumps(monitors[0]))
            with self.assertRaisesRegex(
                confirm.ConfirmError, "disagrees with raw log",
            ):
                confirm.analyze_monitors(scout_path, monitor_paths)


if __name__ == "__main__":
    unittest.main()
