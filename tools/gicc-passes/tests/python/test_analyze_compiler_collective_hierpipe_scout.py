import json
import sys
import tempfile
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PASS_ROOT / "experiments" / "collective"))

import analyze_compiler_collective_hierpipe_scout as scout


class CompilerCollectiveHierpipeScoutTests(unittest.TestCase):
    def test_mixed_pipeline_winners_and_headroom_pass_gate(self):
        rows = {
            scout.ARMS[0]: {
                str(size): 50.0 if index < 3 else 200.0
                for index, size in enumerate(scout.SIZES)
            },
            scout.ARMS[1]: {str(size): 100.0 for size in scout.SIZES},
            scout.ARMS[2]: {
                str(size): 50.0 if index >= 6 else 200.0
                for index, size in enumerate(scout.SIZES)
            },
        }
        result = scout.analyze_rows(rows)
        self.assertTrue(result["hierarchical_pipeline_gate"]["passed"])
        self.assertEqual(3, len(result["aggregate"]["distinct_size_winners"]))
        self.assertGreater(
            result["aggregate"]["best_uniform_over_pointwise_geomean"], 1.05,
        )

    def test_uniform_pipeline_winner_fails_gate(self):
        rows = {
            scout.ARMS[0]: {str(size): 100.0 for size in scout.SIZES},
            scout.ARMS[1]: {str(size): 99.0 for size in scout.SIZES},
            scout.ARMS[2]: {str(size): 101.0 for size in scout.SIZES},
        }
        result = scout.analyze_rows(rows)
        self.assertFalse(result["hierarchical_pipeline_gate"]["passed"])
        self.assertEqual(
            [scout.ARMS[1]], result["aggregate"]["distinct_size_winners"],
        )
        with self.assertRaisesRegex(scout.ScoutError, "size coverage"):
            scout.analyze_rows({**rows, scout.ARMS[2]: {"1024": 90.0}})

    def test_monitor_archive_is_exact_and_content_addressed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            output = root / "output"
            bundle.mkdir()
            output.mkdir()
            freeze = bundle / "FROZEN_V3_MANIFEST.json"
            runner = root / "run_compiler_collective_hierpipe_scout.sh"
            freeze.write_text("{}\n")
            runner.write_text("#!/usr/bin/env bash\n")
            driver = output / "driver.out"
            lines = [
                "HIERPIPE_CONFIG nodes=4 ranks=32 ppn=8 runs=3 warmup=1 "
                "arms=" + " ".join(scout.ARMS),
            ]
            for arm in scout.ARMS:
                lines.extend([
                    f"HIERPIPE_ARM_START arm={arm}",
                    f"HIERPIPE_ARM_DONE arm={arm}",
                ])
            lines.append("HIERPIPE_DONE")
            driver.write_text("\n".join(lines) + "\n")
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
                "schema_version": "gicc-collective-replicate-job-monitor-v1",
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
                    "duration_seconds": 1200.0,
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

            monitor["jobspec"]["duration_seconds"] = 1201.0
            monitor_path.write_text(json.dumps(monitor))
            with self.assertRaisesRegex(
                scout.ScoutError, "exactly four nodes",
            ):
                scout.analyze_monitor(monitor_path)


if __name__ == "__main__":
    unittest.main()
