; HIP still represents the launch through a compiler device stub at the
; pre-inliner extension point. The guarded transform must audit and wrap that
; stub dispatch so every later inlined callsite inherits the guard.
;
; REQUIRES: gicc_lowering
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/k_guarded_early_meta.json \
; RUN:    %t.metadir/_Z9k_guarded.json
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_guarded_early_dwq.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-guarded-early-trigger-host,verify' \
; RUN:          -S %s 2>%t.err | %FileCheck %s --check-prefix=STUB
; RUN: %FileCheck %s --check-prefix=LOG < %t.err

target triple = "x86_64-unknown-linux-gnu"

@.str.gicc = private unnamed_addr constant [17 x i8] c"gicc.launch_site\00", section "llvm.metadata"
@.str.f = private unnamed_addr constant [12 x i8] c"launch.hpp\00\00", section "llvm.metadata"
@_Z9k_guarded = constant ptr @_Z25__device_stub__k_guarded, align 8

@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }]
  [{ ptr, ptr, ptr, i32, ptr }
   { ptr @_ZN4gicc6launchIXadL_Z9k_guardedEEEv,
     ptr @.str.gicc, ptr @.str.f, i32 47, ptr null }],
  section "llvm.metadata"

declare i32 @__hipPushCallConfiguration(i64, i32, i64, i32, i64, ptr)
declare i32 @__hipPopCallConfiguration(ptr, ptr, ptr, ptr)
declare i32 @hipLaunchKernel(ptr, i64, i32, i64, i32, ptr, i64, ptr)
declare ptr @runtime_prepare(ptr)

define void @_Z25__device_stub__k_guarded(
    ptr %ctx, ptr %source, ptr %out, i32 %peer, i32 %buf, i64 %size) {
entry:
  %ctx.arg = alloca ptr, align 8
  %source.arg = alloca ptr, align 8
  %out.arg = alloca ptr, align 8
  %peer.arg = alloca i32, align 4
  %buf.arg = alloca i32, align 4
  %size.arg = alloca i64, align 8
  %params = alloca [6 x ptr], align 8
  %grid = alloca i64, align 8
  %block = alloca i64, align 8
  %shmem = alloca i64, align 8
  %stream = alloca ptr, align 8
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
  call i32 @__hipPopCallConfiguration(
      ptr %grid, ptr %block, ptr %shmem, ptr %stream)
  %grid.xy = load i64, ptr %grid, align 8
  %block.xy = load i64, ptr %block, align 8
  %shmem.value = load i64, ptr %shmem, align 8
  %stream.value = load ptr, ptr %stream, align 8
  call i32 @hipLaunchKernel(
      ptr @_Z9k_guarded, i64 %grid.xy, i32 1,
      i64 %block.xy, i32 1, ptr %params,
      i64 %shmem.value, ptr %stream.value)
  ret void
}

define linkonce_odr void @_ZN4gicc6launchIXadL_Z9k_guardedEEEv(
    ptr %rt, i64 %grid.xy, i32 %grid.z, i64 %block.xy, i32 %block.z,
    ptr %source, ptr %out, i32 %peer, i32 %buf, i64 %size) {
entry:
  %ctx = call ptr @runtime_prepare(ptr %rt)
  call i32 @__hipPushCallConfiguration(
      i64 %grid.xy, i32 %grid.z, i64 %block.xy, i32 %block.z,
      i64 0, ptr null)
  %stub = load ptr, ptr @_Z9k_guarded, align 8
  call void %stub(ptr %ctx, ptr %source, ptr %out,
                  i32 %peer, i32 %buf, i64 %size)
  ret void
}

define void @main(ptr %rt, ptr %source, ptr %out,
                  i32 %peer, i32 %buf, i64 %size) {
entry:
  call void @_ZN4gicc6launchIXadL_Z9k_guardedEEEv(
      ptr %rt, i64 1, i32 1, i64 1, i32 1,
      ptr %source, ptr %out, i32 %peer, i32 %buf, i64 %size)
  call void @_ZN4gicc6launchIXadL_Z9k_guardedEEEv(
      ptr %rt, i64 1, i32 1, i64 1, i32 1,
      ptr %source, ptr %out, i32 %peer, i32 %buf, i64 %size)
  ret void
}

; STUB-LABEL: define linkonce_odr void @_ZN4gicc6launchIXadL_Z9k_guardedEEEv(
; STUB: %gicc.guard.params = alloca ptr, i32 6
; STUB: call i32 @__hipPushCallConfiguration
; STUB: call i32 @gicc_runtime_kernel_arg_matches_local_buffer(ptr %rt, ptr %gicc.guard.params, i32 1, i32 4)
; STUB: call i32 @gicc_runtime_local_buffer_contains_interval(ptr %rt, ptr %gicc.guard.params, i32 4, i64 0, i64 {{%.*}})
; STUB: call i32 @gicc_runtime_local_buffer_disjoint_from_kernel_arg_allocation(ptr %rt, ptr %gicc.guard.params, i32 4, i32 2)
; STUB: br i1 {{%.*}}, label %gicc.early.guarded, label %gicc.early.original
; STUB: gicc.early.guarded:
; STUB-NEXT: call void @gicc_runtime_set_schedule_phase_from_kernel_args(ptr %gicc.guard.params, i32 3, ptr null)
; STUB: call void %stub(
; STUB: call void @gicc_runtime_set_schedule_phase_from_kernel_args(ptr %gicc.guard.params, i32 0, ptr null)
; STUB: gicc.early.original:
; STUB: call void %stub(
; STUB-LABEL: define void @main(
; STUB-COUNT-2: call void @_ZN4gicc6launchIXadL_Z9k_guardedEEEv(
; LOG: [guarded-early-trigger-host] _Z9k_guarded: materialized allocation-guarded single launch
