; Device LTO independently rebuilds the producer facts, partitions the exact
; single-entry compute region by checked byte overlap, and phase-gates the
; original communication group.  Discovery writes the input metadata in the
; same opt invocation; fission does not trust in-memory analysis state.
;
; REQUIRES: gicc_lowering
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_fission_dwq.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-device-discovery,gicc-producer-fission-device,verify' \
; RUN:          -S %s 2>%t.err | %FileCheck %s --check-prefix=PHASE
; RUN: %FileCheck %s --check-prefix=LOG < %t.err
;
; The synthetic remainder-only flush is hidden from source-site numbering but
; must still be consumed by ordinary device lowering.  Both it and the
; original-only completion become conditional MMIO triggers.
; RUN: rm -rf %t.lower.metadir && mkdir -p %t.lower.metadir
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.lower.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_fission_dwq.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-device-discovery,gicc-producer-fission-device,gicc-device-lowering,verify' \
; RUN:          -S %s | %FileCheck %s --check-prefix=LOWER

target triple = "amdgcn-amd-amdhsa"

declare void @_ZN4gicc3putEPN4gicc9DeviceCtxEiimimmi(
    ptr, i32, i32, i64, i32, i64, i64, i32)
declare void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr)
declare void @_ZN4gicc5quietEPN4gicc9DeviceCtxEii(ptr, i32, i32)

define amdgpu_kernel void @k_partition(
    ptr addrspace(1) %ctx, ptr addrspace(1) %out, i32 %buf,
    i64 %top, i64 %bottom, i64 %size, i64 %idx, i64 %limit) {
entry:
  %ctx.generic = addrspacecast ptr addrspace(1) %ctx to ptr
  call void @_ZN4gicc3putEPN4gicc9DeviceCtxEiimimmi(
      ptr %ctx.generic, i32 1, i32 %buf, i64 0,
      i32 %buf, i64 %top, i64 %size, i32 0)
  call void @_ZN4gicc3putEPN4gicc9DeviceCtxEiimimmi(
      ptr %ctx.generic, i32 2, i32 %buf, i64 0,
      i32 %buf, i64 %bottom, i64 %size, i32 0)
  %active = icmp ult i64 %idx, %limit
  br i1 %active, label %compute, label %merge

compute:
  %store.ptr = getelementptr i8, ptr addrspace(1) %out, i64 %idx
  store i8 7, ptr addrspace(1) %store.ptr, align 1
  br label %merge

merge:
  call void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr %ctx.generic)
  call void @_ZN4gicc5quietEPN4gicc9DeviceCtxEii(
      ptr %ctx.generic, i32 0, i32 3)
  ret void
}

; PHASE-LABEL: define amdgpu_kernel void @k_partition(
; PHASE: [[PHASEPTR:%.*]] = getelementptr i8, ptr addrspace(1) %ctx, i64 16
; PHASE: [[PHASEVAL:%.*]] = load i32, ptr addrspace(1) [[PHASEPTR]], align 4
; PHASE: [[PRODUCER:%.*]] = icmp eq i32 [[PHASEVAL]], 1
; PHASE: [[REMAINDER:%.*]] = icmp eq i32 [[PHASEVAL]], 2
; PHASE: %gicc.phase.communication = or i1 %gicc.phase.original, [[REMAINDER]]
; PHASE: br i1 %gicc.phase.communication, label %gicc.fission.transfer.do, label %gicc.fission.transfer.cont
; PHASE: br i1 %gicc.phase.communication, label %gicc.fission.transfer.do{{[0-9]*}}, label %gicc.fission.transfer.cont{{[0-9]*}}
; PHASE: br i1 [[REMAINDER]], label %gicc.fission.early_flush.do, label %gicc.fission.early_flush.cont
; PHASE: call void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr %ctx.generic), !gicc.producer_fission.synthetic_flush
; PHASE: [[TOPEND:%.*]] = add i64 %top, %size
; PHASE: icmp ult i64 %idx, [[TOPEND]]
; PHASE: [[BOTTOMEND:%.*]] = add i64 %bottom, %size
; PHASE: icmp ult i64 %idx, [[BOTTOMEND]]
; PHASE: [[SELECTED:%.*]] = and i1 %active, {{%.*}}
; PHASE: br i1 [[SELECTED]], label %compute, label %merge
; PHASE: br i1 {{%.*}}, label %gicc.fission.original_flush.do, label %gicc.fission.original_flush.cont
; PHASE: br i1 %gicc.phase.communication, label %gicc.fission.quiet.do, label %gicc.fission.quiet.cont
; LOG: [producer-fission-device] k_partition: materialized exact producer/remainder partition

; LOWER-LABEL: define amdgpu_kernel void @k_partition(
; LOWER-NOT: call void @_ZN4gicc3putE
; LOWER-COUNT-2: store volatile i64
; LOWER-NOT: call void @_ZN4gicc5flushE
; LOWER-NOT: call void @_ZN4gicc5quietE
