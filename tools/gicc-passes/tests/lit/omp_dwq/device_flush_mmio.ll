; REQUIRES: gicc_lowering
; GICCDeviceLowering in omp-dwq mode: gicc::omp_dwq::flush (the .internalized
; fastcc variant) must be replaced by the same lead-thread MMIO trigger store
; as the HIP gicc::flush path.
;
; The mangled name _ZN4gicc7omp_dwq5flushEPNS_9DeviceCtxE.internalized contains
; the "5flushE" token that classifyGICCCall matches, so no extra matcher is
; needed for omp_dwq flush.
;
; The put marker (_ZN4gicc7omp_dwq3putEPNS_9DeviceCtxEiimimm.internalized)
; contains "3putE" which classifyGICCCall also matches (as PutNoDb).  With no
; hint supplied, proxyAware defaults to false and the put call is erased --
; the host trace owns the work.
;
; RUN: env GICC_MODE=omp-dwq \
; RUN:   %opt -load-pass-plugin=%gicc_passes_so \
; RUN:        -passes='gicc-device-lowering' \
; RUN:        -S %s | %FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

declare fastcc void @_ZN4gicc7omp_dwq3putEPNS_9DeviceCtxEiimimm.internalized(ptr, i32, i32, i64, i32, i64, i64)
declare fastcc void @_ZN4gicc7omp_dwq5flushEPNS_9DeviceCtxE.internalized(ptr)
declare i32 @__kmpc_target_init(ptr, ptr)
declare void @__kmpc_target_deinit()

define amdgpu_kernel void @__omp_offloading_14_7363c44__Z3runPN4gicc9DeviceCtxEiimimm_l3(
    ptr noalias noundef %dyn_env, ptr noundef %ctx,
    i64 noundef %peer_ext, i64 noundef %dbuf_ext, i64 noundef %doff,
    i64 noundef %sbuf_ext, i64 noundef %soff, i64 noundef %n) {
entry:
  %init = call i32 @__kmpc_target_init(ptr null, ptr %dyn_env)
  %is_main = icmp eq i32 %init, -1
  br i1 %is_main, label %body, label %exit

body:
  %peer    = trunc i64 %peer_ext to i32
  %dst_buf = trunc i64 %dbuf_ext to i32
  %src_buf = trunc i64 %sbuf_ext to i32
  tail call fastcc void @_ZN4gicc7omp_dwq3putEPNS_9DeviceCtxEiimimm.internalized(
      ptr noundef %ctx, i32 noundef %peer, i32 noundef %dst_buf, i64 noundef %doff,
      i32 noundef %src_buf, i64 noundef %soff, i64 noundef %n)
  tail call fastcc void @_ZN4gicc7omp_dwq5flushEPNS_9DeviceCtxE.internalized(
      ptr noundef %ctx)
  call void @__kmpc_target_deinit()
  br label %exit

exit:
  ret void
}

; The put marker is erased (proxyAware=false, no hint).
; CHECK-NOT: call fastcc void @_ZN4gicc7omp_dwq3put

; The flush is lowered to the lead-thread MMIO trigger pattern.
; CHECK-LABEL: define amdgpu_kernel void @__omp_offloading_14_7363c44
; CHECK: call i32 @llvm.amdgcn.workitem.id.x()
; CHECK: call i32 @llvm.amdgcn.workgroup.id.x()
; CHECK: getelementptr i8, ptr %ctx, i64 0
; CHECK: load ptr, ptr %{{.*}}, align 8
; CHECK: getelementptr i8, ptr %ctx, i64 8
; CHECK: load i64, ptr %{{.*}}, align 8
; CHECK: store volatile i64
; CHECK: fence release
; CHECK-NOT: call fastcc void @_ZN4gicc7omp_dwq5flush
