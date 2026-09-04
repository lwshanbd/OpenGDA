import json
import tempfile
import unittest
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = (
    ROOT / "tools" / "gicc-passes" / "experiments" / "guarded_early_trigger"
)
sys.path.insert(0, str(EXPERIMENT))

import analyze_guarded_early_trigger_scout as analyzer
import audit_guarded_early_trigger_ir as ir_auditor
import monitor_guarded_early_trigger_scout as monitor


class GuardedEarlyTriggerScoutTests(unittest.TestCase):
    def write_run(
        self, directory: Path, size: int, arm: str, delta: float = 0.0
    ) -> None:
        stdout = []
        for run in range(10):
            suffix = " (warmup)" if run < 2 else ""
            stdout.append(f"Run {run}: {100.0 + run + delta} us{suffix}")
        stdout.append(f"gicc::launch average (runs 2-9): {105.5 + delta} us")
        (directory / "run.out").write_text("\n".join(stdout) + "\n")
        bytes_per_rank = size * (size // 16) * 4
        launches = 161 if arm == "baseline" else 483
        stderr = [
            f"GICC_MM_CHECKSUM rank={rank} bytes={bytes_per_rank} "
            f"fnv64={rank:016x} launches={launches}"
            for rank in range(16)
        ]
        (directory / "run.err").write_text("\n".join(stderr) + "\n")

    def test_parse_run_checks_timing_and_all_rank_checksums(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_run(directory, 4096, "baseline")
            result = monitor.parse_run(
                directory / "run.out", directory / "run.err", 4096,
                "baseline",
            )
            self.assertEqual(105.5, result["measured_mean_us"])
            self.assertEqual(16, len(result["checksums"]))
            self.assertEqual(
                {161}, set(result["successful_kernel_launches"].values())
            )
            (directory / "run.err").write_text(
                (directory / "run.err").read_text().replace("rank=15", "rank=14")
            )
            with self.assertRaisesRegex(monitor.common.MonitorError, "duplicate"):
                monitor.parse_run(
                    directory / "run.out", directory / "run.err", 4096,
                    "baseline",
                )

    def test_parse_run_rejects_silent_guard_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_run(directory, 8192, "guarded")
            error = directory / "run.err"
            error.write_text(
                error.read_text().replace("launches=483", "launches=161")
            )
            with self.assertRaisesRegex(monitor.common.MonitorError, "launch counts"):
                monitor.parse_run(
                    directory / "run.out", error, 8192, "guarded"
                )

    def test_analyzer_applies_preregistered_headroom_gate(self):
        value = {"state": "passed", "runs": {}}
        for replicate, speedup in enumerate((1.03, 1.04, 1.05, 0.99), 1):
            value["runs"][str(replicate)] = {
                "4096": {"speedup": speedup},
                "8192": {"speedup": 0.98},
            }
        result = analyzer.analyze(value)
        self.assertTrue(result["correctness_gate"]["passed"])
        self.assertEqual(
            [4096], result["oracle_headroom_gate"]["promising_sizes"]
        )
        self.assertFalse(result["oracle_headroom_gate"]["paper_claim"])

    def test_final_ir_auditor_accepts_only_guarded_shape(self):
        kernel = ir_auditor.KERNEL
        stub = ir_auditor.STUB
        wrapper = "_ZN4gicc6launchABCmatmul_step_kernelXYZ"
        device = f"""define amdgpu_kernel void @{kernel}() {{
  %gicc.phase = load i32, ptr null
  %x = icmp ne i32 %gicc.phase, 3
gicc.early.synthetic_flush.do:
  store volatile i64 1, ptr null
  %a = atomicrmw fadd ptr null, float 1.0 monotonic
  store volatile i64 1, ptr null
gicc.early.original_flush.cont:
  ret void
}}
"""
        host = f"""define void @{stub}() {{
  call i32 @hipLaunchKernel(ptr @{kernel})
  ret void
}}
define void @{wrapper}() {{
  call i32 @__hipPushCallConfiguration()
  call i32 @gicc_runtime_kernel_arg_matches_local_buffer()
  call i32 @gicc_runtime_kernel_arg_matches_local_buffer()
  call i32 @gicc_runtime_local_buffer_contains_interval()
  call i32 @gicc_runtime_local_buffer_disjoint_from_kernel_arg_allocation()
gicc.early.guarded:
  call void @gicc_runtime_set_schedule_phase_from_kernel_args(ptr null, i32 3, ptr null)
  call i32 @hipLaunchKernel(ptr @{kernel})
  call void @gicc_runtime_set_schedule_phase_from_kernel_args(ptr null, i32 0, ptr null)
gicc.early.original:
  call i32 @hipLaunchKernel(ptr @{kernel})
  ret void
}}
define i32 @main() {{
  call void @{wrapper}()
  call void @{wrapper}()
  ret i32 0
}}
"""
        self.assertEqual(2, ir_auditor.audit_device(device)["trigger_stores"])
        self.assertEqual(2, ir_auditor.audit_host(host)["guarded_kernel_launches"])
        with self.assertRaisesRegex(ir_auditor.AuditError, "straddle"):
            ir_auditor.audit_device(device.replace(
                "gicc.early.synthetic_flush.do:\n"
                "  store volatile i64 1, ptr null\n"
                "  %a = atomicrmw fadd ptr null, float 1.0 monotonic\n",
                "gicc.early.synthetic_flush.do:\n"
                "  %a = atomicrmw fadd ptr null, float 1.0 monotonic\n"
                "  store volatile i64 1, ptr null\n",
            ))

    def test_controller_is_serial_pdebug_and_model_free(self):
        controller = EXPERIMENT / "continue_guarded_early_trigger_scout.sh"
        subprocess.run(["bash", "-n", controller], check=True)
        text = controller.read_text(encoding="utf-8")
        self.assertIn("waiting_predecessor", text)
        self.assertIn("flux batch -q pdebug", text)
        self.assertNotIn("-q pci", text)
        self.assertNotIn("anthropic", text.lower())
        self.assertNotIn("model_trials", text)


if __name__ == "__main__":
    unittest.main()
