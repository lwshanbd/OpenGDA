; A PHI with three incoming values (e.g., from three different control-
; flow paths) is non-canonical and TraceTemplateBuilder cannot handle
; it. HK should reject so the trace function isn't silently wrong.
;
; RUN: env GICC_MODE=feature-extract \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-hk-analysis' \
; RUN:          -disable-output %s 2>&1 | %FileCheck %s

target triple = "amdgcn-amd-amdhsa"

declare void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(ptr, i32, i32, i64, i32, i64, i64)

define amdgpu_kernel void @three_phi(ptr %ctx, i32 %peer, i32 %sel) {
entry:
  %c0 = icmp eq i32 %sel, 0
  br i1 %c0, label %a, label %b
a:
  br label %end
b:
  %c1 = icmp eq i32 %sel, 1
  br i1 %c1, label %b1, label %b2
b1:
  br label %end
b2:
  br label %end
end:
  ; 3-way merge — not a canonical loop iv.
  %v = phi i64 [ 1, %a ], [ 2, %b1 ], [ 3, %b2 ]
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %peer, i32 0, i64 0, i32 0, i64 0, i64 %v)
  ret void
}

; CHECK: error: gicc::put_no_db: argument 'size' is not host-knowable
; CHECK: note: PHI has != 2 incoming values
