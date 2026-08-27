; REQUIRES: gicc_lowering
; The registered source is an integer buffer handle, so LLVM AA cannot map a
; pointer store back to it.  Fail closed on any intervening memory write.
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: not env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_device_group_early.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-device-lowering' -disable-output %s 2>&1 | \
; RUN:     %FileCheck %s

target triple = "amdgcn-amd-amdhsa"

declare void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
    ptr, i32, i32, i64, i32, i64, i64)
declare void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr)

define amdgpu_kernel void @kernel_group(
    ptr %ctx, ptr addrspace(1) %out, i32 %top, i32 %bottom, i32 %buf) {
entry:
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %top, i32 %buf, i64 0,
      i32 %buf, i64 0, i64 4096)
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %bottom, i32 %buf, i64 4096,
      i32 %buf, i64 4096, i64 4096)
  store volatile i32 1, ptr addrspace(1) %out
  call void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr %ctx)
  ret void
}

; CHECK: LLVM ERROR: gicc: TRIGGER_GROUP_EARLY rejected for group ?:?:kernel_group::0: intervening instruction may write a registered source buffer
