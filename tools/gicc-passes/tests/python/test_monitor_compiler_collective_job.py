import sys
import tempfile
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PASS_ROOT / "experiments" / "collective"))

import monitor_compiler_collective_job as monitor


class CompilerCollectiveJobMonitorTests(unittest.TestCase):
    def test_validates_complete_zero_error_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "job.out"
            path.write_text(
                "COLLECTIVE_CONFIG plan=baseline ranks=16 ppn=8 runs=1 warmup=0\n"
                "RESULT plan=baseline nodes=2 ranks=16 ppn=8 bytes=4096 "
                "median_us=12.5 errors=0\n"
                "COLLECTIVE_DONE plan=baseline total_errors=0\n"
            )
            record = monitor.validate_output(
                path, "baseline", [4096], 2, 16, 8, 1, 0
            )
            self.assertEqual({"4096": 12.5}, record["results"])
            self.assertEqual(0, record["total_errors"])

    def test_rejects_incomplete_or_incorrect_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "job.out"
            path.write_text(
                "COLLECTIVE_CONFIG plan=baseline ranks=16 ppn=8 runs=1 warmup=0\n"
                "RESULT plan=baseline nodes=2 ranks=16 ppn=8 bytes=4096 "
                "median_us=12.5 errors=1\n"
            )
            with self.assertRaisesRegex(monitor.MonitorError, "correctness errors"):
                monitor.validate_output(
                    path, "baseline", [4096], 2, 16, 8, 1, 0
                )

    def test_scheduler_timeout_is_a_failure(self):
        events = [
            {"name": "exception", "context": {"type": "timeout"}},
            {"name": "finish", "context": {"status": 36352}},
            {"name": "clean", "context": {}},
        ]
        result = monitor.scheduler_result(events)
        self.assertEqual(142, result["exit_code"])
        self.assertEqual(["timeout"], result["exception_types"])

    def test_jobspec_requires_pdebug_and_redacts_environment(self):
        jobspec = {
            "tasks": [{"command": ["/tmp/program", "1", "0"]}],
            "resources": [{"type": "node", "count": 2}],
            "attributes": {"system": {
                "queue": "pdebug",
                "cwd": "/tmp",
                "duration": 180.0,
                "environment": {
                    "GICC_COLL_SIZES": "4096",
                    "SECRET_TOKEN": "must-not-be-recorded",
                },
            }},
        }
        summary = monitor.summarize_jobspec(jobspec)
        self.assertEqual("pdebug", summary["queue"])
        self.assertEqual({"GICC_COLL_SIZES": "4096"}, summary["environment"])
        jobspec["attributes"]["system"]["queue"] = "pci"
        with self.assertRaisesRegex(monitor.MonitorError, "expected 'pdebug'"):
            monitor.summarize_jobspec(jobspec)


if __name__ == "__main__":
    unittest.main()
