import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PASS_ROOT / "experiments" / "collective"))

import monitor_compiler_collective_job as monitor
import monitor_compiler_collective_replicate as replicate_monitor


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

    def test_stderr_summary_keeps_last_hdir_checkpoint_per_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "job.err"
            path.write_text(
                "[gicc] CPU proxy: 1 worker thread(s) per rank\n"
                "[hdir r1 call0 4096 B] 2-launch-k1\n"
                "[hdir r0 call0 4096 B] 3-sync-k1-enter\n"
                "[hdir r1 call0 4096 B] 3-sync-k1-enter\n"
            )
            summary = monitor.summarize_stderr(path, tail_lines=2)
            self.assertEqual(4, summary["line_count"])
            self.assertEqual(2, len(summary["tail"]))
            self.assertEqual(
                "3-sync-k1-enter", summary["hdir_last_by_rank"]["0"]["stage"]
            )
            self.assertEqual(
                "3-sync-k1-enter", summary["hdir_last_by_rank"]["1"]["stage"]
            )

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
                "files": {"script": {"data": "#!/bin/sh\necho batch\n"}},
            }},
        }
        summary = monitor.summarize_jobspec(jobspec)
        self.assertEqual("pdebug", summary["queue"])
        self.assertEqual({"GICC_COLL_SIZES": "4096"}, summary["environment"])
        self.assertEqual(
            hashlib.sha256(b"#!/bin/sh\necho batch\n").hexdigest(),
            summary["embedded_script_sha256"],
        )
        jobspec["attributes"]["system"]["queue"] = "pci"
        with self.assertRaisesRegex(monitor.MonitorError, "expected 'pdebug'"):
            monitor.summarize_jobspec(jobspec)

    def test_resource_summary_freezes_exact_pdebug_nodes(self):
        resources = {
            "version": 1,
            "execution": {
                "R_lite": [{
                    "rank": "29-30",
                    "children": {"core": "0-63", "gpu": "0-7"},
                }],
                "nodelist": ["tioga[38-39]"],
                "properties": {"pall": "29-30", "pdebug": "29-30"},
                "starttime": 10,
                "expiration": 20,
            },
        }
        summary = monitor.summarize_resources(resources)
        self.assertEqual(["tioga[38-39]"], summary["nodelist"])
        self.assertEqual(["29-30"], summary["ranks"])
        resources["execution"]["properties"].pop("pdebug")
        with self.assertRaisesRegex(monitor.MonitorError, "pdebug allocation"):
            monitor.summarize_resources(resources)

    def test_replicate_monitor_requires_unique_named_log_paths(self):
        values = [
            replicate_monitor.named_path("baseline_auto=/tmp/base.out"),
            replicate_monitor.named_path("hierarchical_ring=/tmp/hier.out"),
        ]
        result = replicate_monitor.unique_paths(values, "benchmark")
        self.assertEqual(
            {"baseline_auto", "hierarchical_ring"}, set(result)
        )
        with self.assertRaisesRegex(monitor.MonitorError, "duplicate"):
            replicate_monitor.unique_paths([values[0], values[0]], "benchmark")


if __name__ == "__main__":
    unittest.main()
