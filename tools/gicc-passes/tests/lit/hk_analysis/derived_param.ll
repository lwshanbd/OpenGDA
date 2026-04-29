; The 'size' argument is the sum of two kernel parameters — HK pure
; arithmetic over kernel formals → accepted.
;
; RUN: env GICC_MODE=feature-extract \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-hk-analysis' \
; RUN:          -disable-output %s 2>&1 | %FileCheck %s --allow-empty

target triple = "amdgcn-amd-amdhsa"

declare void @_ZN4gicc10put_no_dbEPN4gicc9DeviceCtxEiimimm(ptr, i32, i32, i64, i32, i64, i64)

define amdgpu_kernel void @sum_kernel(ptr %ctx, i32 %peer, i64 %a, i64 %b) {
entry:
  %sum = add i64 %a, %b
  call void @_ZN4gicc10put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %peer, i32 0, i64 0, i32 0, i64 0, i64 %sum)
  ret void
}

; CHECK-NOT: error: gicc::
