import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "tools/gicc-passes/experiments/analyze_lto_trace_retrospective.py"
SPEC = importlib.util.spec_from_file_location("lto_trace_retrospective", SCRIPT)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def log_text(arm: str, batch: int, scale: float = 1.0) -> str:
    header = (f"=== bench_pingpong_lto (LTO-generated host trace) ===\n"
              f"ranks=2 outer_iters=21 batch={batch} warmup=10\n")
    if arm == "handwritten":
        header = ("=== bench_pingpong (mode=dwq) ===\n"
                  f"(batch override: batch_per_outer={batch})\n")
    rows = "\n".join(
        f"{size} 84 {10.0 * scale:.3f} {10.0 * scale:.3f}"
        for size in audit.SIZES
    )
    expected = 31 * batch * len(audit.SIZES)
    if arm == "lto":
        trailer = (f"[enqueue-audit] mono_total_ops={expected} "
                   f"expected={expected} match=YES\n")
    else:
        trailer = (f"[enqueue-audit mode=dwq] mono_total_ops={expected + 64} "
                   f"dwq_expected={expected} match=MORE\n")
    return header + rows + "\n" + trailer


class LtoTraceRetrospectiveTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path]:
        logs = root / "logs"
        logs.mkdir()
        driver = root / "run_alloc.sh"
        driver.write_text(
            "for trial in $(seq 1 10); do\n"
            "run_one $trial handwritten 4 ./h x\n"
            "run_one $trial lto         4 ./l x\n"
            "run_one $trial handwritten 64 ./h x\n"
            "run_one $trial lto         64 ./l x\n"
            "done\n"
        )
        for batch in audit.BATCHES:
            for trial in audit.TRIALS:
                (logs / f"trial_{trial}_handwritten_b{batch}.log").write_text(
                    log_text("handwritten", batch, 1.0)
                )
                (logs / f"trial_{trial}_lto_b{batch}.log").write_text(
                    log_text("lto", batch, 0.8)
                )
        return logs, driver

    def test_strict_complete_analysis_keeps_claims_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs, driver = self.make_fixture(Path(tmp))
            result = audit.analyze(logs, driver)
            self.assertEqual(result["coverage"]["raw_logs"], 40)
            self.assertAlmostEqual(result["results"]["4"]["aggregate_geomean"],
                                   1.25)
            self.assertTrue(result["claim_gate"]
                            ["compiler_generated_trace_executed"])
            self.assertFalse(result["claim_gate"]
                             ["confirmatory_performance_equivalence"])
            self.assertFalse(result["claim_gate"]["llm_performance_measured"])

    def test_missing_log_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs, driver = self.make_fixture(Path(tmp))
            (logs / "trial_10_lto_b64.log").unlink()
            with self.assertRaises(audit.AuditError):
                audit.analyze(logs, driver)

    def test_lto_counter_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs, driver = self.make_fixture(Path(tmp))
            path = logs / "trial_1_lto_b4.log"
            path.write_text(path.read_text().replace("match=YES", "match=LESS"))
            with self.assertRaises(audit.AuditError):
                audit.analyze(logs, driver)


if __name__ == "__main__":
    unittest.main()
