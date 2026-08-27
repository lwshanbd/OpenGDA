; REQUIRES: gicc_lowering
; A candidate ID cannot move a trigger when the selected put is not actually
; inside a device natural loop.  The build must fail closed.
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: not env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_device_coalesce_early_bad.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-device-lowering' -disable-output %s 2>&1 | \
; RUN:     %FileCheck %s

target triple = "amdgcn-amd-amdhsa"

declare void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
    ptr, i32, i32, i64, i32, i64, i64)
declare void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr)

define amdgpu_kernel void @kernel_early_bad(ptr %ctx, i32 %peer, i32 %buf) {
entry:
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %peer, i32 %buf, i64 0,
      i32 %buf, i64 0, i64 4096)
  call void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr %ctx)
  ret void
}

; CHECK: LLVM ERROR: gicc: COALESCE_LOOP_EARLY rejected for site ?:?:kernel_early_bad::0: put_no_db is not inside a natural loop
