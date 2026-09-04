; A pointer write without a complete noalias root proof must keep the original
; trigger schedule even when the hint requests GUARDED_EARLY_TRIGGER.
;
; REQUIRES: gicc_lowering
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_guarded_early_dwq.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-device-discovery,gicc-guarded-early-trigger-device,verify' \
; RUN:          -S %s 2>%t.err | %FileCheck %s --check-prefix=SAFE
; RUN: %FileCheck %s --check-prefix=LOG < %t.err
; RUN: %FileCheck %s --check-prefix=META \
; RUN:     < %t.metadir/k_guarded_alias.json

target triple = "amdgcn-amd-amdhsa"

declare void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
    ptr, i32, i32, i64, i32, i64, i64)
declare void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr)
declare void @_ZN4gicc5quietEPN4gicc9DeviceCtxEii(ptr, i32, i32)

define amdgpu_kernel void @k_guarded_alias(
    ptr addrspace(1) %ctx, ptr addrspace(1) noalias readonly %source,
    ptr addrspace(1) %out, i32 %peer, i32 %buf, i64 %size) {
entry:
  %ctx.generic = addrspacecast ptr addrspace(1) %ctx to ptr
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx.generic, i32 %peer, i32 %buf, i64 0,
      i32 %buf, i64 0, i64 %size)
  %source.value = load float, ptr addrspace(1) %source, align 4
  %old = atomicrmw fadd ptr addrspace(1) %out, float %source.value monotonic
  call void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr %ctx.generic)
  call void @_ZN4gicc5quietEPN4gicc9DeviceCtxEii(
      ptr %ctx.generic, i32 0, i32 3)
  ret void
}

; SAFE-LABEL: define amdgpu_kernel void @k_guarded_alias(
; SAFE-NOT: gicc.phase
; SAFE-COUNT-1: call void @_ZN4gicc9put_no_dbE
; SAFE-COUNT-1: atomicrmw fadd
; SAFE-COUNT-1: call void @_ZN4gicc5flushE
; SAFE-COUNT-1: call void @_ZN4gicc5quietE
; LOG: [guarded-early-trigger-device] k_guarded_alias: rejected: final compiler facts disagree with persisted metadata: guarded early-trigger proof is absent or unsafe
; META-NOT: "guarded_early_trigger_device_materialized"
; META: "guarded_early_trigger_guardable": false
