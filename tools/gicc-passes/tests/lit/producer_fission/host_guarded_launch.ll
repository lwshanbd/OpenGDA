; The dormant host half of producer-frontier fission must re-prove the final
; HIP launch ABI and emit a fail-closed runtime guard.  It is available only
; by explicit pass name until the matching device-body phase partition exists.
;
; REQUIRES: gicc_lowering
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/k_overlap_partition_meta.json \
; RUN:    %t.metadir/_Z11k_partition.json
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_fission_dwq.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-producer-fission-host' -S %s 2>%t.err | \
; RUN:     %FileCheck %s --check-prefix=PHASED
; RUN: %FileCheck %s --check-prefix=LOG < %t.err
;
; An opposite phase-sensitive guard destroys the shared disabling value.  The
; pass must preserve exactly the original launch and add no runtime helpers.
; RUN: rm -rf %t.reject.metadir && mkdir -p %t.reject.metadir
; RUN: python3 %S/../Inputs/add_phase_guard_mismatch.py \
; RUN:   %S/../Inputs/k_overlap_partition_meta.json \
; RUN:   %t.reject.metadir/_Z11k_partition.json
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.reject.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_fission_dwq.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-producer-fission-host' -S %s 2>%t.reject.err | \
; RUN:     %FileCheck %s --check-prefix=REJECT
; RUN: %FileCheck %s --check-prefix=REJECT-LOG < %t.reject.err
;
; An eager IPC route would read the halo source before the producer phase.
; Even otherwise-valid compiler facts must therefore keep the fused launch.
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_fission_ipc.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-producer-fission-host' -S %s 2>%t.route.err | \
; RUN:     %FileCheck %s --check-prefix=ROUTE
; RUN: %FileCheck %s --check-prefix=ROUTE-LOG < %t.route.err

target triple = "x86_64-unknown-linux-gnu"

@.str.gicc = private unnamed_addr constant [17 x i8] c"gicc.launch_site\00", section "llvm.metadata"
@.str.f = private unnamed_addr constant [12 x i8] c"launch.hpp\00\00", section "llvm.metadata"

@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }]
  [{ ptr, ptr, ptr, i32, ptr }
   { ptr @_ZN4gicc6launchIXadL_Z11k_partitionEEEv,
     ptr @.str.gicc, ptr @.str.f, i32 47, ptr null }],
  section "llvm.metadata"

declare void @_Z11k_partition()
declare i32 @hipLaunchKernel(ptr, i64, i32, i64, i32, ptr, i64, ptr)

define linkonce_odr void @_ZN4gicc6launchIXadL_Z11k_partitionEEEv(
    ptr %rt, i64 %grid.xy, i32 %grid.z, i64 %block.xy, i32 %block.z,
    i64 %shmem, ptr %stream, ptr %ctx, ptr %out, i32 %buf,
    i64 %top, i64 %bottom, i64 %size, i64 %stride,
    i1 %calculate_norm) {
entry:
  %ctx.arg = alloca ptr, align 8
  %out.arg = alloca ptr, align 8
  %buf.arg = alloca i32, align 4
  %top.arg = alloca i64, align 8
  %bottom.arg = alloca i64, align 8
  %size.arg = alloca i64, align 8
  %stride.arg = alloca i64, align 8
  %norm.arg = alloca i8, align 1
  %params = alloca [8 x ptr], align 8
  store ptr %ctx, ptr %ctx.arg, align 8
  store ptr %out, ptr %out.arg, align 8
  store i32 %buf, ptr %buf.arg, align 4
  store i64 %top, ptr %top.arg, align 8
  store i64 %bottom, ptr %bottom.arg, align 8
  store i64 %size, ptr %size.arg, align 8
  store i64 %stride, ptr %stride.arg, align 8
  %norm.abi = zext i1 %calculate_norm to i8
  store i8 %norm.abi, ptr %norm.arg, align 1
  %slot0 = getelementptr inbounds [8 x ptr], ptr %params, i64 0, i64 0
  %slot1 = getelementptr inbounds [8 x ptr], ptr %params, i64 0, i64 1
  %slot2 = getelementptr inbounds [8 x ptr], ptr %params, i64 0, i64 2
  %slot3 = getelementptr inbounds [8 x ptr], ptr %params, i64 0, i64 3
  %slot4 = getelementptr inbounds [8 x ptr], ptr %params, i64 0, i64 4
  %slot5 = getelementptr inbounds [8 x ptr], ptr %params, i64 0, i64 5
  %slot6 = getelementptr inbounds [8 x ptr], ptr %params, i64 0, i64 6
  %slot7 = getelementptr inbounds [8 x ptr], ptr %params, i64 0, i64 7
  store ptr %ctx.arg, ptr %slot0, align 8
  store ptr %out.arg, ptr %slot1, align 8
  store ptr %buf.arg, ptr %slot2, align 8
  store ptr %top.arg, ptr %slot3, align 8
  store ptr %bottom.arg, ptr %slot4, align 8
  store ptr %size.arg, ptr %slot5, align 8
  store ptr %stride.arg, ptr %slot6, align 8
  store ptr %norm.arg, ptr %slot7, align 8
  %rc = call i32 @hipLaunchKernel(
      ptr @_Z11k_partition,
      i64 %grid.xy, i32 %grid.z, i64 %block.xy, i32 %block.z,
      ptr %params, i64 %shmem, ptr %stream)
  ret void
}

define void @main(ptr %rt, ptr %stream, ptr %ctx, ptr %out, i32 %buf,
                  i64 %top, i64 %bottom, i64 %size, i64 %stride,
                  i1 %calculate_norm) {
entry:
  call void @_ZN4gicc6launchIXadL_Z11k_partitionEEEv(
      ptr %rt, i64 1, i32 1, i64 1, i32 1, i64 0, ptr %stream,
      ptr %ctx, ptr %out, i32 %buf, i64 %top, i64 %bottom,
      i64 %size, i64 %stride, i1 %calculate_norm)
  ret void
}

; PHASED-LABEL: define linkonce_odr void @_ZN4gicc6launchIXadL_Z11k_partitionEEEv(
; PHASED: call i32 @gicc_runtime_kernel_arg_matches_local_buffer(ptr %rt, ptr %params, i32 1, i32 2)
; PHASED: [[TOP:%.*]] = load i64, ptr %top.arg
; PHASED: [[SIZE0:%.*]] = load i64, ptr %size.arg
; PHASED: call i32 @gicc_runtime_local_buffer_contains_interval(ptr %rt, ptr %params, i32 2, i64 [[TOP]], i64 [[SIZE0]])
; PHASED: [[BOTTOM:%.*]] = load i64, ptr %bottom.arg
; PHASED: [[SIZE1:%.*]] = load i64, ptr %size.arg
; PHASED: call i32 @gicc_runtime_local_buffer_contains_interval(ptr %rt, ptr %params, i32 2, i64 [[BOTTOM]], i64 [[SIZE1]])
; PHASED: [[NORM:%.*]] = load i8, ptr %norm.arg
; PHASED: [[SIDE:%.*]] = icmp eq i8 [[NORM]], 0
; PHASED: br i1 {{%.*}}, label %gicc.fission.phased, label %gicc.fission.fused
; PHASED: gicc.fission.phased:
; PHASED-NEXT: call void @gicc_runtime_set_schedule_phase_from_kernel_args(ptr %params, i32 1, ptr %stream)
; PHASED-NEXT: {{%.*}} = call i32 @hipLaunchKernel(ptr @_Z11k_partition
; PHASED: call void @gicc_runtime_set_schedule_phase_from_kernel_args(ptr %params, i32 2, ptr %stream)
; PHASED-NEXT: {{%.*}} = call i32 @hipLaunchKernel(ptr @_Z11k_partition
; PHASED: call void @gicc_runtime_set_schedule_phase_from_kernel_args(ptr %params, i32 0, ptr %stream)
; PHASED-NEXT: br label %gicc.fission.cont
; PHASED: gicc.fission.fused:
; PHASED-NEXT: {{%.*}} = call i32 @hipLaunchKernel(ptr @_Z11k_partition
; LOG: [producer-fission-host] _Z11k_partition: materialized guarded two-phase launch

; REJECT-LABEL: define linkonce_odr void @_ZN4gicc6launchIXadL_Z11k_partitionEEEv(
; REJECT-NOT: @gicc_runtime_kernel_arg_matches_local_buffer
; REJECT-NOT: @gicc_runtime_local_buffer_contains_interval
; REJECT-NOT: @gicc_runtime_set_schedule_phase_from_kernel_args
; REJECT-COUNT-1: call i32 @hipLaunchKernel(
; REJECT: ret void
; REJECT-LOG: [producer-fission-host] _Z11k_partition: rejected: non-duplicable operations have no shared disabling guard

; ROUTE-LABEL: define linkonce_odr void @_ZN4gicc6launchIXadL_Z11k_partitionEEEv(
; ROUTE-NOT: @gicc_runtime_kernel_arg_matches_local_buffer
; ROUTE-COUNT-1: call i32 @hipLaunchKernel(
; ROUTE: ret void
; ROUTE-LOG: [producer-fission-host] _Z11k_partition: rejected: every fission transfer must use untransformed DWQ_TRIGGER
