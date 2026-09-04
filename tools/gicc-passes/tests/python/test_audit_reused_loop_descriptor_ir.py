import importlib.util
import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    PASS_ROOT / "experiments" / "reused_loop_descriptor"
    / "audit_reused_loop_descriptor_ir.py"
)
SPEC = importlib.util.spec_from_file_location("audit_reused_descriptor", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


BASELINE_HOST = """
define internal void @gicc_trace_dwq_loop_kernel(ptr %rt) {
  %dwq.peers = alloca i32, i32 4
  %dwq.dst_bufs = alloca i32, i32 4
  %dwq.dst_offs = alloca i64, i32 4
  %dwq.src_bufs = alloca i32, i32 4
  %dwq.src_offs = alloca i64, i32 4
  %dwq.sizes = alloca i64, i32 4
  call void @gicc_runtime_dwq_enqueue_batched(ptr %rt, i32 4, ptr %dwq.peers,
      ptr %dwq.dst_bufs, ptr %dwq.dst_offs, ptr %dwq.src_bufs,
      ptr %dwq.src_offs, ptr %dwq.sizes)
  ret void
}
define void @main() {
  call i32 @hipLaunchKernel(ptr @_Z15dwq_loop_kernelPN4gicc9DeviceCtxEiimi)
  ret void
}
"""

REUSED_HOST = """
define internal void @gicc_trace_dwq_loop_kernel(ptr %rt, i32 %n) {
  %positive = icmp sgt i32 %n, 0
  %count = select i1 %positive, i32 %n, i32 0
  call void @gicc_runtime_dwq_enqueue_repeated(ptr %rt, i32 %count,
      i32 1, i32 2, i64 0, i32 2, i64 0, i64 8)
  ret void
}
define void @main() {
  call i32 @hipLaunchKernel(ptr @_Z15dwq_loop_kernelPN4gicc9DeviceCtxEiimi)
  ret void
}
"""

DEVICE = f"""
define amdgpu_kernel void @{audit.KERNEL}(ptr %ctx) {{
  store volatile i64 1, ptr addrspace(1) null
  ret void
}}
"""


class ReusedLoopDescriptorAuditTests(unittest.TestCase):
    def test_two_frozen_host_shapes(self):
        baseline = audit.audit_host(BASELINE_HOST, "baseline")
        reused = audit.audit_host(REUSED_HOST, "reused")
        self.assertEqual(1, baseline["batched_helper_calls"])
        self.assertEqual(1, reused["repeated_helper_calls"])
        self.assertEqual(0, reused["descriptor_array_name_occurrences"])

    def test_reused_trace_may_be_inlined(self):
        inlined = REUSED_HOST.replace(
            "define internal void @gicc_trace_dwq_loop_kernel",
            "define internal void @launch_owner",
        )
        result = audit.audit_host(inlined, "reused")
        self.assertIsNone(result["retained_trace"])

    def test_device_shape_is_one_original_trigger(self):
        result = audit.audit_device(DEVICE)
        self.assertEqual(1, result["trigger_stores"])

    def test_reused_arm_rejects_descriptor_arrays(self):
        with self.assertRaisesRegex(audit.AuditError, "descriptor arrays"):
            audit.audit_host(
                REUSED_HOST.replace(
                    "  ret void\n}",
                    "  %dwq.peers = alloca i32, i32 1\n  ret void\n}",
                    1,
                ),
                "reused",
            )

    def test_device_rejects_extra_trigger(self):
        bad = DEVICE.replace(
            "  ret void", "  store volatile i64 2, ptr addrspace(1) null\n  ret void"
        )
        with self.assertRaisesRegex(audit.AuditError, "exactly one"):
            audit.audit_device(bad)


if __name__ == "__main__":
    unittest.main()
