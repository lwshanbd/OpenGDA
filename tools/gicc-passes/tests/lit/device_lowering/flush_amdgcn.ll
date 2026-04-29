; GICCDeviceLowering: gicc::flush is replaced by a lead-thread MMIO
; trigger store + agent-system release fence on AMDGCN.
;
; RUN: env GICC_MODE=lower \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-device-lowering' \
; RUN:          -S %s | %FileCheck %s

target triple = "amdgcn-amd-amdhsa"

declare void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr)

define amdgpu_kernel void @k(ptr %ctx) {
  call void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr %ctx)
  ret void
}

; CHECK-LABEL: define amdgpu_kernel void @k
; CHECK: call i32 @llvm.amdgcn.workitem.id.x()
; CHECK: call i32 @llvm.amdgcn.workgroup.id.x()
; CHECK: call ptr @gicc_runtime_trigger_addr(ptr %ctx)
; CHECK: call i64 @gicc_runtime_trigger_val(ptr %ctx)
; CHECK: store volatile i64
; CHECK: fence syncscope("agent-system") release
; CHECK-NOT: call void @_ZN4gicc5flush
