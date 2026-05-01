; The 'size' argument is a PHI feeding from a loop counter that loads
; from device memory each iteration — non-HK in v1 (per-thread loop
; bound). HK pass surfaces a warning so the dispatch lowering routes
; this site to CPU_PROXY_ENQUEUE.
;
; RUN: env GICC_MODE=feature-extract \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-hk-analysis' \
; RUN:          -disable-output %s 2>&1 | %FileCheck %s

target triple = "amdgcn-amd-amdhsa"

declare void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(ptr, i32, i32, i64, i32, i64, i64)

define amdgpu_kernel void @loopy(ptr %ctx, i32 %peer, ptr %src) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %next, %loop ]
  ; load from device memory feeds the iv update — guarantees non-HK
  %v   = load i64, ptr %src, align 8
  %next = add i64 %iv, %v
  %done = icmp uge i64 %next, 8
  br i1 %done, label %exit, label %loop

exit:
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %peer, i32 0, i64 0, i32 0, i64 0, i64 %next)
  ret void
}

; CHECK: warning: gicc::put_no_db: argument 'size' is not host-knowable
; CHECK-SAME: CPU_PROXY_ENQUEUE
