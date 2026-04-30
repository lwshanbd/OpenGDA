; GICCDeviceLowering: put_no_db calls become no-ops in device IR
; (the host trace function takes over). The kernel must still terminate
; with `ret void` and contain no remaining gicc::put_no_db symbol.
;
; RUN: env GICC_MODE=lower \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-device-lowering' \
; RUN:          -S %s | %FileCheck %s

target triple = "amdgcn-amd-amdhsa"

declare void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(ptr, i32, i32, i64, i32, i64, i64)

define amdgpu_kernel void @k(ptr %ctx, i32 %p, i32 %b, i64 %off, i64 %sz) {
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %p, i32 %b, i64 %off, i32 %b, i64 %off, i64 %sz)
  ret void
}

; CHECK-LABEL: define amdgpu_kernel void @k
; CHECK-NOT: call void @_ZN4gicc9put_no_db
; CHECK: ret void
