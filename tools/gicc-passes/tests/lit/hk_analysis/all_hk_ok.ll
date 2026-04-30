; All gicc::put_no_db arguments are kernel formal parameters → HK passes
; with no error output.
;
; RUN: env GICC_MODE=feature-extract \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-hk-analysis' \
; RUN:          -disable-output %s 2>&1 | %FileCheck %s --allow-empty

target triple = "amdgcn-amd-amdhsa"

declare void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(ptr, i32, i32, i64, i32, i64, i64)

define amdgpu_kernel void @halo_kernel(ptr %ctx, i32 %peer, i32 %dst_buf,
                                       i64 %dst_off, i32 %src_buf,
                                       i64 %src_off, i64 %size) {
entry:
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %peer, i32 %dst_buf, i64 %dst_off,
      i32 %src_buf, i64 %src_off, i64 %size)
  ret void
}

; CHECK-NOT: error: gicc::
