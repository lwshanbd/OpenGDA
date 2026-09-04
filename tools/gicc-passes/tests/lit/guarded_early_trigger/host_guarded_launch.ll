; Final host LTO must require the device attestation and emit a full runtime
; identity/bounds/allocation guard. The true and false edges each launch the
; unchanged kernel exactly once.
;
; REQUIRES: gicc_lowering
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/k_guarded_early_meta.json \
; RUN:    %t.metadir/_Z9k_guarded.json
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_guarded_early_dwq.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-guarded-early-trigger-host,verify' \
; RUN:          -S %s 2>%t.err | %FileCheck %s --check-prefix=GUARDED
; RUN: %FileCheck %s --check-prefix=LOG < %t.err
;
; A route-only hint cannot change trigger placement.
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_guarded_early_unselected.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-guarded-early-trigger-host,verify' \
; RUN:          -S %s 2>%t.unselected.err | \
; RUN:     %FileCheck %s --check-prefix=UNSELECTED
; RUN: %FileCheck %s --check-prefix=UNSELECTED-LOG < %t.unselected.err
;
; A selected transform without final-device attestation remains one launch.
; RUN: sed '/guarded_early_trigger_device_materialized/d' \
; RUN:     %S/../Inputs/k_guarded_early_meta.json \
; RUN:     > %t.metadir/_Z9k_guarded.json
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_guarded_early_dwq.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-guarded-early-trigger-host,verify' \
; RUN:          -S %s 2>%t.noattest.err | \
; RUN:     %FileCheck %s --check-prefix=NOATTEST
; RUN: %FileCheck %s --check-prefix=NOATTEST-LOG < %t.noattest.err

target triple = "x86_64-unknown-linux-gnu"

@.str.gicc = private unnamed_addr constant [17 x i8] c"gicc.launch_site\00", section "llvm.metadata"
@.str.f = private unnamed_addr constant [12 x i8] c"launch.hpp\00\00", section "llvm.metadata"

@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }]
  [{ ptr, ptr, ptr, i32, ptr }
   { ptr @_ZN4gicc6launchIXadL_Z9k_guardedEEEv,
     ptr @.str.gicc, ptr @.str.f, i32 47, ptr null }],
  section "llvm.metadata"

declare void @_Z9k_guarded()
declare i32 @hipLaunchKernel(ptr, i64, i32, i64, i32, ptr, i64, ptr)

define linkonce_odr void @_ZN4gicc6launchIXadL_Z9k_guardedEEEv(
    ptr %rt, i64 %grid.xy, i32 %grid.z, i64 %block.xy, i32 %block.z,
    i64 %shmem, ptr %stream, ptr %ctx, ptr %source, ptr %out,
    i32 %peer, i32 %buf, i64 %size) {
entry:
  %ctx.arg = alloca ptr, align 8
  %source.arg = alloca ptr, align 8
  %out.arg = alloca ptr, align 8
  %peer.arg = alloca i32, align 4
  %buf.arg = alloca i32, align 4
  %size.arg = alloca i64, align 8
  %params = alloca [6 x ptr], align 8
  store ptr %ctx, ptr %ctx.arg, align 8
  store ptr %source, ptr %source.arg, align 8
  store ptr %out, ptr %out.arg, align 8
  store i32 %peer, ptr %peer.arg, align 4
  store i32 %buf, ptr %buf.arg, align 4
  store i64 %size, ptr %size.arg, align 8
  %slot0 = getelementptr inbounds [6 x ptr], ptr %params, i64 0, i64 0
  %slot1 = getelementptr inbounds [6 x ptr], ptr %params, i64 0, i64 1
  %slot2 = getelementptr inbounds [6 x ptr], ptr %params, i64 0, i64 2
  %slot3 = getelementptr inbounds [6 x ptr], ptr %params, i64 0, i64 3
  %slot4 = getelementptr inbounds [6 x ptr], ptr %params, i64 0, i64 4
  %slot5 = getelementptr inbounds [6 x ptr], ptr %params, i64 0, i64 5
  store ptr %ctx.arg, ptr %slot0, align 8
  store ptr %source.arg, ptr %slot1, align 8
  store ptr %out.arg, ptr %slot2, align 8
  store ptr %peer.arg, ptr %slot3, align 8
  store ptr %buf.arg, ptr %slot4, align 8
  store ptr %size.arg, ptr %slot5, align 8
  %rc = call i32 @hipLaunchKernel(
      ptr @_Z9k_guarded,
      i64 %grid.xy, i32 %grid.z, i64 %block.xy, i32 %block.z,
      ptr %params, i64 %shmem, ptr %stream)
  ret void
}

define void @main(ptr %rt, ptr %stream, ptr %ctx, ptr %source, ptr %out,
                  i32 %peer, i32 %buf, i64 %size) {
entry:
  call void @_ZN4gicc6launchIXadL_Z9k_guardedEEEv(
      ptr %rt, i64 1, i32 1, i64 1, i32 1, i64 0, ptr %stream,
      ptr %ctx, ptr %source, ptr %out, i32 %peer, i32 %buf, i64 %size)
  ret void
}

; GUARDED-LABEL: define linkonce_odr void @_ZN4gicc6launchIXadL_Z9k_guardedEEEv(
; GUARDED: call i32 @gicc_runtime_kernel_arg_matches_local_buffer(ptr %rt, ptr %params, i32 1, i32 4)
; GUARDED: [[SIZE:%.*]] = load i64, ptr %size.arg
; GUARDED: call i32 @gicc_runtime_local_buffer_contains_interval(ptr %rt, ptr %params, i32 4, i64 0, i64 [[SIZE]])
; GUARDED: call i32 @gicc_runtime_local_buffer_disjoint_from_kernel_arg_allocation(ptr %rt, ptr %params, i32 4, i32 2)
; GUARDED: br i1 {{%.*}}, label %gicc.early.guarded, label %gicc.early.original
; GUARDED: gicc.early.guarded:
; GUARDED-NEXT: call void @gicc_runtime_set_schedule_phase_from_kernel_args(ptr %params, i32 3, ptr %stream)
; GUARDED-NEXT: {{%.*}} = call i32 @hipLaunchKernel(ptr @_Z9k_guarded
; GUARDED: call void @gicc_runtime_set_schedule_phase_from_kernel_args(ptr %params, i32 0, ptr %stream)
; GUARDED-NEXT: br label %gicc.early.cont
; GUARDED: gicc.early.original:
; GUARDED-NEXT: {{%.*}} = call i32 @hipLaunchKernel(ptr @_Z9k_guarded
; LOG: [guarded-early-trigger-host] _Z9k_guarded: materialized allocation-guarded single launch

; UNSELECTED-LABEL: define linkonce_odr void @_ZN4gicc6launchIXadL_Z9k_guardedEEEv(
; UNSELECTED-NOT: @gicc_runtime_kernel_arg_matches_local_buffer
; UNSELECTED-NOT: @gicc_runtime_local_buffer_contains_interval
; UNSELECTED-NOT: @gicc_runtime_local_buffer_disjoint_from_kernel_arg_allocation
; UNSELECTED-NOT: @gicc_runtime_set_schedule_phase_from_kernel_args
; UNSELECTED-COUNT-1: call i32 @hipLaunchKernel(
; UNSELECTED: ret void
; UNSELECTED-LOG: [guarded-early-trigger-host] _Z9k_guarded: rejected: every guarded early PUT must request the compiler-owned GUARDED_EARLY_TRIGGER transform

; NOATTEST-LABEL: define linkonce_odr void @_ZN4gicc6launchIXadL_Z9k_guardedEEEv(
; NOATTEST-NOT: @gicc_runtime_kernel_arg_matches_local_buffer
; NOATTEST-NOT: @gicc_runtime_set_schedule_phase_from_kernel_args
; NOATTEST-COUNT-1: call i32 @hipLaunchKernel(
; NOATTEST: ret void
; NOATTEST-LOG: [guarded-early-trigger-host] _Z9k_guarded: rejected: final device LTO did not attest the early/original trigger partition
