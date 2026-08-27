; REQUIRES: gicc_lowering
; COALESCE_LOOP_EARLY must not trust a placement assertion from the hint.  The
; device pass identifies the selected put site, proves that it is the kernel's
; only communication site inside a natural loop with a unique exit, proves one
; later flush dominated by that exit, and only then moves/lower the MMIO store.
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_device_coalesce_early.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-device-lowering' -S %s | %FileCheck %s

target triple = "amdgcn-amd-amdhsa"

declare void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
    ptr, i32, i32, i64, i32, i64, i64)
declare void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr)

define amdgpu_kernel void @kernel_early(ptr %ctx, i32 %peer, i32 %buf) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %next, %loop ]
  %off = mul i64 %iv, 4096
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %peer, i32 %buf, i64 %off,
      i32 %buf, i64 %off, i64 4096)
  %next = add nuw nsw i64 %iv, 1
  %more = icmp ult i64 %next, 16
  br i1 %more, label %loop, label %after

after:
  %compute = fadd float 1.000000e+00, 2.000000e+00
  call void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr %ctx)
  store volatile float %compute, ptr addrspace(1) null
  ret void
}

; CHECK-LABEL: define amdgpu_kernel void @kernel_early
; CHECK-LABEL: after:
; CHECK: call i32 @llvm.amdgcn.workitem.id.x()
; CHECK: br i1 {{.*}}, label %flush.do, label %flush.skip
; CHECK-LABEL: flush.do:
; CHECK: store volatile i64 {{.*}}, ptr {{.*}}, align 8, !nontemporal !{{[0-9]+}}, !gicc.communication_transform ![[EARLY:[0-9]+]]
; CHECK-LABEL: flush.skip:
; CHECK: %compute = fadd float
; CHECK-NOT: call void @_ZN4gicc9put_no_db
; CHECK: ![[EARLY]] = !{!"COALESCE_LOOP_EARLY"}
