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
; CHECK: getelementptr i8, ptr %ctx, i64 0
; CHECK: load ptr, ptr %{{.*}}, align 8
; CHECK: getelementptr i8, ptr %ctx, i64 16
; CHECK: load i64, ptr %{{.*}}, align 8
; CHECK: store volatile i64
; CHECK: fence release
; CHECK-NOT: call void @_ZN4gicc5flush
