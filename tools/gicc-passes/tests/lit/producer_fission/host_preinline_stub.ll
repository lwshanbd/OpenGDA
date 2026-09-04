; Producer fission must wrap the compiler HIP stub before O3 can inline an
; unguarded launch into each application caller. The guarded edge dispatches
; producer and remainder phases; the fallback edge dispatches once.
;
; REQUIRES: gicc_lowering
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/k_overlap_partition_meta.json \
; RUN:    %t.metadir/_Z11k_partition.json
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_fission_dwq.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-producer-fission-host,verify' \
; RUN:          -S %s 2>%t.err | %FileCheck %s --check-prefix=STUB
; RUN: %FileCheck %s --check-prefix=LOG < %t.err

target triple = "x86_64-unknown-linux-gnu"

@.str.gicc = private unnamed_addr constant [17 x i8] c"gicc.launch_site\00", section "llvm.metadata"
@.str.f = private unnamed_addr constant [12 x i8] c"launch.hpp\00\00", section "llvm.metadata"
@_Z11k_partition = constant ptr @_Z29__device_stub__k_partition, align 8

@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }]
  [{ ptr, ptr, ptr, i32, ptr }
   { ptr @_ZN4gicc6launchIXadL_Z11k_partitionEEEv,
     ptr @.str.gicc, ptr @.str.f, i32 47, ptr null }],
  section "llvm.metadata"

declare i32 @__hipPushCallConfiguration(i64, i32, i64, i32, i64, ptr)
declare i32 @__hipPopCallConfiguration(ptr, ptr, ptr, ptr)
declare i32 @hipLaunchKernel(ptr, i64, i32, i64, i32, ptr, i64, ptr)
declare ptr @runtime_prepare(ptr)

define void @_Z29__device_stub__k_partition(
    ptr %ctx, ptr %out, i32 %buf, i64 %top, i64 %bottom, i64 %size,
    i64 %stride, i1 %calculate_norm) {
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
  %grid = alloca i64, align 8
  %block = alloca i64, align 8
  %shmem = alloca i64, align 8
  %stream = alloca ptr, align 8
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
  call i32 @__hipPopCallConfiguration(
      ptr %grid, ptr %block, ptr %shmem, ptr %stream)
  %grid.xy = load i64, ptr %grid, align 8
  %block.xy = load i64, ptr %block, align 8
  %shmem.value = load i64, ptr %shmem, align 8
  %stream.value = load ptr, ptr %stream, align 8
  call i32 @hipLaunchKernel(
      ptr @_Z11k_partition, i64 %grid.xy, i32 1,
      i64 %block.xy, i32 1, ptr %params,
      i64 %shmem.value, ptr %stream.value)
  ret void
}

define linkonce_odr void @_ZN4gicc6launchIXadL_Z11k_partitionEEEv(
    ptr %rt, i64 %grid.xy, i32 %grid.z, i64 %block.xy, i32 %block.z,
    ptr %out, i32 %buf, i64 %top, i64 %bottom, i64 %size,
    i64 %stride, i1 %calculate_norm) {
entry:
  %ctx = call ptr @runtime_prepare(ptr %rt)
  call i32 @__hipPushCallConfiguration(
      i64 %grid.xy, i32 %grid.z, i64 %block.xy, i32 %block.z,
      i64 0, ptr null)
  %stub = load ptr, ptr @_Z11k_partition, align 8
  call void %stub(ptr %ctx, ptr %out, i32 %buf, i64 %top,
                  i64 %bottom, i64 %size, i64 %stride,
                  i1 %calculate_norm)
  ret void
}

define void @main(ptr %rt, ptr %out, i32 %buf, i64 %top, i64 %bottom,
                  i64 %size, i64 %stride, i1 %calculate_norm) {
entry:
  call void @_ZN4gicc6launchIXadL_Z11k_partitionEEEv(
      ptr %rt, i64 1, i32 1, i64 1, i32 1, ptr %out, i32 %buf,
      i64 %top, i64 %bottom, i64 %size, i64 %stride,
      i1 %calculate_norm)
  ret void
}

; STUB-LABEL: define linkonce_odr void @_ZN4gicc6launchIXadL_Z11k_partitionEEEv(
; STUB: %gicc.guard.params = alloca ptr, i32 8
; STUB: call i32 @__hipPushCallConfiguration
; STUB: call i32 @gicc_runtime_kernel_arg_matches_local_buffer(ptr %rt, ptr %gicc.guard.params, i32 1, i32 2)
; STUB: call i32 @gicc_runtime_local_buffer_contains_interval(ptr %rt, ptr %gicc.guard.params, i32 2, i64 {{%.*}}, i64 {{%.*}})
; STUB: br i1 {{%.*}}, label %gicc.fission.phased, label %gicc.fission.fused
; STUB: gicc.fission.phased:
; STUB-NEXT: call void @gicc_runtime_set_schedule_phase_from_kernel_args(ptr %gicc.guard.params, i32 1, ptr null)
; STUB: call void %stub(
; STUB: call void @gicc_runtime_set_schedule_phase_from_kernel_args(ptr %gicc.guard.params, i32 2, ptr null)
; STUB: call void %stub(
; STUB: call void @gicc_runtime_set_schedule_phase_from_kernel_args(ptr %gicc.guard.params, i32 0, ptr null)
; STUB: gicc.fission.fused:
; STUB: call void %stub(
; LOG: [producer-fission-host] _Z11k_partition: materialized guarded two-phase launch
