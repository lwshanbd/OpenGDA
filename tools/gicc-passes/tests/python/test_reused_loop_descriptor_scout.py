import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = PASS_ROOT / "experiments" / "reused_loop_descriptor"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


monitor = load_module(
    "monitor_reused_loop_descriptor_scout",
    EXPERIMENT / "monitor_reused_loop_descriptor_scout.py",
)
analyzer = load_module(
    "analyze_reused_loop_descriptor_scout",
    EXPERIMENT / "analyze_reused_loop_descriptor_scout.py",
)


def valid_stdout(batch, speed=10.0):
    lines = [
        "=== bench_pingpong_lto (LTO-generated host trace) ===",
        f"ranks=2  outer_iters=21  batch={batch}  warmup=10",
    ]
    for size in monitor.SIZES:
        lines.append(f"{size} {21 * batch} {speed + 0.5:.3f} {speed:.3f}")
    expected = 31 * batch * len(monitor.SIZES)
    lines.append(
        f"[enqueue-audit] mono_total_ops={expected} expected={expected} match=YES"
    )
    return "\n".join(lines) + "\n"


def synthetic_monitor(speedup):
    runs = {}
    for replicate in monitor.REPLICATES:
        batches = {}
        for batch in monitor.BATCHES:
            expected = 31 * batch * len(monitor.SIZES)
            batches[str(batch)] = {
                "baseline": {
                    "enqueue_actual": expected,
                    "enqueue_expected": expected,
                },
                "reused": {
                    "enqueue_actual": expected,
                    "enqueue_expected": expected,
                },
                "per_size_speedup": {
                    size: speedup for size in monitor.SIZES
                },
            }
        runs[str(replicate)] = batches
    return {"state": "passed", "runs": runs}


class ReusedLoopDescriptorScoutTests(unittest.TestCase):
    def test_parse_run_requires_all_rows_and_exact_enqueue_count(self):
        with tempfile.TemporaryDirectory() as directory:
            stdout = Path(directory) / "run.out"
            stderr = Path(directory) / "run.err"
            stdout.write_text(valid_stdout(4), encoding="utf-8")
            stderr.write_text("", encoding="utf-8")
            result = monitor.parse_run(stdout, stderr, 4)
            self.assertEqual(1984, result["enqueue_actual"])
            self.assertEqual(list(monitor.SIZES), list(result["rows"]))

            stdout.write_text(
                valid_stdout(4).replace("mono_total_ops=1984", "mono_total_ops=1983"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(monitor.common.MonitorError, "enqueue audit"):
                monitor.parse_run(stdout, stderr, 4)

    def test_parse_run_rejects_failure_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            stdout = Path(directory) / "run.out"
            stderr = Path(directory) / "run.err"
            stdout.write_text(valid_stdout(64), encoding="utf-8")
            stderr.write_text("VERIFY-FAIL rank=1\n", encoding="utf-8")
            with self.assertRaisesRegex(monitor.common.MonitorError, "failure marker"):
                monitor.parse_run(stdout, stderr, 64)

    def test_allocation_shape_requires_n2_n2_c64_g1(self):
        jobspec = {
            "resources": [{
                "type": "node",
                "count": 2,
                "with": [{
                    "type": "slot",
                    "label": "task",
                    "count": 1,
                    "with": [
                        {"type": "core", "count": 64},
                        {"type": "gpu", "count": 1},
                    ],
                }],
            }],
        }
        resource_set = {"pdebug_ranks": "34-35"}
        monitor.validate_allocation_shape(jobspec, resource_set)
        jobspec["resources"][0]["count"] = 4
        with self.assertRaisesRegex(monitor.common.MonitorError, "not N2"):
            monitor.validate_allocation_shape(jobspec, resource_set)

    def test_analysis_passes_only_frozen_headroom_gate(self):
        passing = analyzer.analyze(synthetic_monitor(1.03))
        self.assertTrue(passing["oracle_headroom_gate"]["passed"])
        self.assertFalse(passing["oracle_headroom_gate"]["paper_claim"])
        self.assertFalse(
            passing["oracle_headroom_gate"]["provider_protocol_permitted"]
        )

        failing = analyzer.analyze(synthetic_monitor(1.00))
        self.assertFalse(failing["oracle_headroom_gate"]["passed"])

        invalid = synthetic_monitor(1.10)
        invalid["runs"]["1"]["4"]["reused"]["enqueue_actual"] -= 1
        with self.assertRaisesRegex(ValueError, "enqueue audit mismatch"):
            analyzer.analyze(invalid)

    def test_analysis_rejects_failed_monitor(self):
        with self.assertRaisesRegex(ValueError, "did not pass"):
            analyzer.analyze({"state": "failed"})


if __name__ == "__main__":
    unittest.main()
