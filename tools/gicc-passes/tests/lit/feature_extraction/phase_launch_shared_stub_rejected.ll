; A device stub used by any kernel dispatch outside the annotated wrapper is
; not a legal rewrite owner: cloning its launch would silently change that
; other source-level launch as well.
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/k_const_size_meta.json \
; RUN:    %t.metadir/_Z9k_const_szPN4gicc9DeviceCtxE.json
; RUN: env GICC_MODE=feature-extract GICC_META_DIR=%t.metadir \
; RUN:                              GICC_FEATURES_OUT=%t.features.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-feature-extraction' \
; RUN:          -disable-output %s
; RUN: cat %t.features.json | %FileCheck %s --check-prefix=JSON

target triple = "x86_64-unknown-linux-gnu"

@.str.gicc = private unnamed_addr constant [17 x i8] c"gicc.launch_site\00", section "llvm.metadata"
@.str.f = private unnamed_addr constant [12 x i8] c"launch.hpp\00\00", section "llvm.metadata"
@_Z9k_const_szPN4gicc9DeviceCtxE = dso_local constant ptr @__device_stub__k_const_sz

@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }]
  [{ ptr, ptr, ptr, i32, ptr }
   { ptr @_ZN4gicc6launchITnDaXadL_Z9k_const_szPN4gicc9DeviceCtxEEEvRNS_7RuntimeE,
     ptr @.str.gicc, ptr @.str.f, i32 47, ptr null }],
  section "llvm.metadata"

declare i32 @__hipPushCallConfiguration(i64, i32, i64, i32, i64, ptr)
declare i32 @__hipPopCallConfiguration(ptr, ptr, ptr, ptr)
declare i32 @hipLaunchKernel(ptr, i64, i32, i64, i32, ptr, i64, ptr)

define linkonce_odr void @__device_stub__k_const_sz(ptr %ctx) {
entry:
  %ctx.arg = alloca ptr, align 8
  %params = alloca [1 x ptr], align 8
  %grid = alloca i64, align 8
  %block = alloca i64, align 8
  %shmem = alloca i64, align 8
  %stream = alloca ptr, align 8
  store ptr %ctx, ptr %ctx.arg, align 8
  store ptr %ctx.arg, ptr %params, align 8
  %pop = call i32 @__hipPopCallConfiguration(
      ptr %grid, ptr %block, ptr %shmem, ptr %stream)
  %s = load ptr, ptr %stream, align 8
  %rc = call i32 @hipLaunchKernel(
      ptr @_Z9k_const_szPN4gicc9DeviceCtxE,
      i64 1, i32 1, i64 1, i32 1, ptr %params, i64 0, ptr %s)
  ret void
}

define linkonce_odr void @_ZN4gicc6launchITnDaXadL_Z9k_const_szPN4gicc9DeviceCtxEEEvRNS_7RuntimeE(
    ptr %rt, i64 %gx, i32 %gz, i64 %bx, i32 %bz,
    i64 %shmem, ptr %stream) {
entry:
  %push = call i32 @__hipPushCallConfiguration(
      i64 %gx, i32 %gz, i64 %bx, i32 %bz, i64 %shmem, ptr %stream)
  %stub = load ptr, ptr @_Z9k_const_szPN4gicc9DeviceCtxE, align 8
  call void %stub(ptr null)
  ret void
}

define void @unannotated_launch() {
entry:
  %stub = load ptr, ptr @_Z9k_const_szPN4gicc9DeviceCtxE, align 8
  call void %stub(ptr null)
  ret void
}

define void @main(ptr %rt, ptr %stream) {
entry:
  call void @_ZN4gicc6launchITnDaXadL_Z9k_const_szPN4gicc9DeviceCtxEEEvRNS_7RuntimeE(
      ptr %rt, i64 1, i32 1, i64 1, i32 1, i64 0, ptr %stream)
  ret void
}

; JSON: "phase_launch_materialization": "none"
; JSON: "phase_launch_reason": "HIP device stub is shared by another kernel dispatch"
; JSON: "phase_launch_stream": "explicit"
; JSON: "phase_launch_supported": false
