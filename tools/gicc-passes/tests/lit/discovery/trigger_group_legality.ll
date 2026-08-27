; Device discovery records a conservative group-level legality proof.  The
; first kernel may move its shared trigger across arithmetic only; the second
; is rejected because a store could update a registered source buffer.
;
; RUN: rm -rf %t.metadir
; RUN: env GICC_MODE=feature-extract GICC_META_DIR=%t.metadir \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-device-discovery' -disable-output %s
; RUN: cat %t.metadir/group_safe.json | \
; RUN:     %FileCheck %s --check-prefix=SAFE
; RUN: cat %t.metadir/group_write.json | \
; RUN:     %FileCheck %s --check-prefix=WRITE

target triple = "amdgcn-amd-amdhsa"

declare void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
    ptr, i32, i32, i64, i32, i64, i64)
declare void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr)

define amdgpu_kernel void @group_safe(
    ptr %ctx, i32 %top, i32 %bottom, i32 %buf) {
entry:
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %top, i32 %buf, i64 0,
      i32 %buf, i64 0, i64 4096)
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %bottom, i32 %buf, i64 4096,
      i32 %buf, i64 4096, i64 4096)
  %compute = fadd float 1.000000e+00, 2.000000e+00
  call void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr %ctx)
  ret void
}

define amdgpu_kernel void @group_write(
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

; SAFE: "completion_site_id": "?:?:group_safe::2"
; SAFE: "group_early_trigger_legal": true
; SAFE: "group_early_trigger_reason": "proved mandatory flush and no intervening memory writes"
; SAFE: "site_id": "?:?:group_safe::0"
; SAFE: "completion_site_id": "?:?:group_safe::2"
; SAFE: "group_early_trigger_legal": true
; SAFE: "site_id": "?:?:group_safe::1"

; WRITE: "completion_site_id": "?:?:group_write::2"
; WRITE: "group_early_trigger_legal": false
; WRITE: "group_early_trigger_reason": "intervening instruction may write a registered source buffer"
; WRITE: "site_id": "?:?:group_write::0"
