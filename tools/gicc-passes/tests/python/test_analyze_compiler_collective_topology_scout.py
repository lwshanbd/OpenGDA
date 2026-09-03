import json
import sys
import tempfile
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PASS_ROOT / "experiments" / "collective"))

import analyze_compiler_collective_topology_scout as scout


class CompilerCollectiveTopologyScoutTests(unittest.TestCase):
    def test_monitor_archive_is_content_addressed_and_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            output = root / "output"
            bundle.mkdir()
            output.mkdir()

            freeze = bundle / "FROZEN_V3_MANIFEST.json"
            runner = root / "run_compiler_collective_topology_scout.sh"
            freeze.write_text("{}\n")
            runner.write_text("#!/usr/bin/env bash\n")
            driver = output / "driver.out"
            driver.write_text(
                "SCOUT_CONFIG nodes=4 ranks=32 ppn=8 runs=3 warmup=1 "
                "arms=baseline_auto hierarchical_double_tree\n"
                "SCOUT_ARM_START arm=baseline_auto\n"
                "SCOUT_ARM_DONE arm=baseline_auto\n"
                "SCOUT_ARM_START arm=hierarchical_double_tree\n"
                "SCOUT_ARM_DONE arm=hierarchical_double_tree\n"
                "SCOUT_DONE\n"
            )
            benchmarks = {}
            for index, arm in enumerate(scout.ARMS):
                log = output / f"{arm}.out"
                log.write_text(f"archive for {arm}\n")
                benchmarks[arm] = {
                    "benchmark": {
                        "results": {
                            str(size): float(100 + index)
                            for size in scout.SIZES
                        },
                    },
                    "stdout": {
                        "path": str(log),
                        "bytes": log.stat().st_size,
                        "sha256": scout.sha256(log),
                    },
                }
            monitor = {
                "schema_version": (
                    "gicc-collective-replicate-job-monitor-v1"
                ),
                "state": "passed",
                "job_id": "job-1",
                "expected": {
                    "nodes": 4, "ranks": 32, "ppn": 8,
                    "runs": 3, "warmup": 1,
                    "sizes": list(scout.SIZES),
                },
                "scheduler": {"exit_code": 0, "exception_types": []},
                "jobspec": {
                    "queue": "pdebug",
                    "duration_seconds": 900.0,
                    "embedded_script_sha256": scout.sha256(runner),
                    "resources": [{
                        "type": "node", "count": 4,
                        "with": [{
                            "type": "slot", "count": 8, "label": "task",
                            "with": [
                                {"type": "core", "count": 8},
                                {"type": "gpu", "count": 1},
                            ],
                        }],
                    }],
                    "command": [
                        "flux", "broker", "-c{{tmpdir}}/conf.json",
                        "{{tmpdir}}/script", str(bundle), str(output),
                    ],
                },
                "resource_set": {"nodelist": ["tioga[1-4]"]},
                "artifacts": [
                    {"path": str(freeze), "sha256": scout.sha256(freeze)},
                    {"path": str(runner), "sha256": scout.sha256(runner)},
                ],
                "driver_stdout": {
                    "path": str(driver),
                    "bytes": driver.stat().st_size,
                    "sha256": scout.sha256(driver),
                },
                "benchmarks": benchmarks,
            }
            monitor_path = root / "monitor.json"
            monitor_path.write_text(json.dumps(monitor))

            result = scout.analyze_monitor(monitor_path)
            self.assertRegex(result["result_id"], r"^sha256:[0-9a-f]{64}$")
            self.assertFalse(result["model_invoked"])
            self.assertEqual(2, result["artifact_count"])

            monitor["jobspec"]["duration_seconds"] = 901.0
            monitor_path.write_text(json.dumps(monitor))
            with self.assertRaisesRegex(
                scout.ScoutError, "did not request exactly four nodes",
            ):
                scout.analyze_monitor(monitor_path)

    def test_promising_topology_requires_broad_and_large_headroom(self):
        baseline = {str(size): 100.0 for size in scout.SIZES}
        tree = {
            str(size): (110.0 if index < 2 else 80.0)
            for index, size in enumerate(scout.SIZES)
        }
        result = scout.analyze_rows({
            "baseline_auto": baseline,
            "hierarchical_double_tree": tree,
        })
        self.assertTrue(result["v4_topology_hypothesis"]["promising"])
        self.assertEqual(
            ["baseline_auto", "hierarchical_double_tree"],
            result["aggregate"]["distinct_size_winners"],
        )

    def test_small_or_uniform_difference_does_not_trigger_v4(self):
        baseline = {str(size): 100.0 for size in scout.SIZES}
        tree = {str(size): 99.0 for size in scout.SIZES}
        result = scout.analyze_rows({
            "baseline_auto": baseline,
            "hierarchical_double_tree": tree,
        })
        self.assertFalse(result["v4_topology_hypothesis"]["promising"])
        with self.assertRaisesRegex(scout.ScoutError, "size coverage"):
            scout.analyze_rows({
                "baseline_auto": baseline,
                "hierarchical_double_tree": {"1024": 80.0},
            })


if __name__ == "__main__":
    unittest.main()
