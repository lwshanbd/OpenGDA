; A collective-only build must not run the independent per-transfer lowering
; pipeline. Without a GICC_HINT_IN/host trace, that lowering would erase this
; proxy put and leave a collective kernel waiting forever for its peer flag.
;
; RUN: env GICC_MODE=lower GICC_COLLECTIVE_ONLY=1 \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='default<O2>' -S %s 2>&1 | %FileCheck %s

target triple = "amdgcn-amd-amdhsa"

declare void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
    ptr, i32, i32, i64, i32, i64, i64)

define amdgpu_kernel void @collective_kernel(
    ptr %ctx, i32 %peer, i32 %buf, i64 %offset, i64 %size) {
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %peer, i32 %buf, i64 %offset,
      i32 %buf, i64 %offset, i64 %size)
  ret void
}

; CHECK: [gicc-pass] mode=lower target=auto scope=collective triple=amdgcn-amd-amdhsa
; CHECK-LABEL: define amdgpu_kernel void @collective_kernel
; CHECK: call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm
; CHECK: ret void
