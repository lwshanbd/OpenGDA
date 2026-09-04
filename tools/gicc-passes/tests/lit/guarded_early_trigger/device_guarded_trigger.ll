; Final device LTO must rediscover the source/write roots and communication
; order before it emits an EARLY_TRIGGER phase. The compute and quiet remain
; exactly once; only the compiler-owned flush is phase-partitioned.
;
; REQUIRES: gicc_lowering
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_guarded_early_dwq.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-device-discovery,gicc-guarded-early-trigger-device,verify' \
; RUN:          -S %s 2>%t.err | %FileCheck %s --check-prefix=PHASE
; RUN: %FileCheck %s --check-prefix=LOG < %t.err
; RUN: %FileCheck %s --check-prefix=ATTEST \
; RUN:     < %t.metadir/_Z9k_guarded.json
;
; RUN: rm -rf %t.lower.metadir && mkdir -p %t.lower.metadir
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.lower.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_guarded_early_dwq.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-device-discovery,gicc-guarded-early-trigger-device,gicc-device-lowering,verify' \
; RUN:          -S %s | %FileCheck %s --check-prefix=LOWER

target triple = "amdgcn-amd-amdhsa"

declare void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
    ptr, i32, i32, i64, i32, i64, i64)
declare void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr)
declare void @_ZN4gicc5quietEPN4gicc9DeviceCtxEii(ptr, i32, i32)

define amdgpu_kernel void @_Z9k_guarded(
    ptr addrspace(1) %ctx, ptr addrspace(1) noalias readonly %source,
    ptr addrspace(1) noalias %out, i32 %peer, i32 %buf, i64 %size) {
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

; PHASE-LABEL: define amdgpu_kernel void @_Z9k_guarded(
; PHASE: [[PHASEPTR:%.*]] = getelementptr i8, ptr addrspace(1) %ctx, i64 16
; PHASE: [[PHASE:%.*]] = load i32, ptr addrspace(1) [[PHASEPTR]], align 4
; PHASE: [[EARLY:%.*]] = icmp eq i32 [[PHASE]], 3
; PHASE-COUNT-1: call void @_ZN4gicc9put_no_dbE
; PHASE: br i1 [[EARLY]], label %gicc.early.synthetic_flush.do, label %gicc.early.synthetic_flush.cont
; PHASE: call void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr %ctx.generic), !gicc.guarded_early_trigger.synthetic_flush
; PHASE-COUNT-1: atomicrmw fadd
; PHASE: br i1 %gicc.phase.original_trigger, label %gicc.early.original_flush.do, label %gicc.early.original_flush.cont
; PHASE: call void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr %ctx.generic)
; PHASE-COUNT-1: call void @_ZN4gicc5quietE
; LOG: [guarded-early-trigger-device] _Z9k_guarded: materialized guarded early/original trigger partition
; ATTEST: "guarded_early_trigger_device_materialized": true

; LOWER-LABEL: define amdgpu_kernel void @_Z9k_guarded(
; LOWER-NOT: call void @_ZN4gicc9put_no_dbE
; LOWER: store volatile i64
; LOWER-COUNT-1: atomicrmw fadd
; LOWER: store volatile i64
; LOWER-NOT: call void @_ZN4gicc5flushE
; LOWER-NOT: call void @_ZN4gicc5quietE
; LOWER: !{{[0-9]+}} = !{!"GUARDED_EARLY_TRIGGER"}
