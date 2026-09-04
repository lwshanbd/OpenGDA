; Verify the real early-simplification HIP shape: the annotated wrapper pushes
; launch configuration and calls through the kernel constant global, whose
; initializer is an exclusive device stub. The reusable parameter array and
; hipLaunchKernel live in that stub rather than the wrapper.
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
  %stream.value = load ptr, ptr %stream, align 8
  %rc = call i32 @hipLaunchKernel(
      ptr @_Z9k_const_szPN4gicc9DeviceCtxE,
      i64 4294967304, i32 1, i64 4294967297, i32 1,
      ptr %params, i64 0, ptr %stream.value)
  ret void
}

define linkonce_odr void @_ZN4gicc6launchITnDaXadL_Z9k_const_szPN4gicc9DeviceCtxEEEvRNS_7RuntimeE(
    ptr %rt, i64 %grid.xy, i32 %grid.z, i64 %block.xy, i32 %block.z,
    i64 %shmem, ptr %stream) {
entry:
  %push = call i32 @__hipPushCallConfiguration(
      i64 %grid.xy, i32 %grid.z, i64 %block.xy, i32 %block.z,
      i64 %shmem, ptr %stream)
  %stub = load ptr, ptr @_Z9k_const_szPN4gicc9DeviceCtxE, align 8
  call void %stub(ptr null)
  ret void
}

define void @main(ptr %rt, ptr %stream) {
entry:
  call void @_ZN4gicc6launchITnDaXadL_Z9k_const_szPN4gicc9DeviceCtxEEEvRNS_7RuntimeE(
      ptr %rt, i64 4294967304, i32 1, i64 4294967297, i32 1,
      i64 0, ptr %stream)
  ret void
}

; JSON: "launch_contexts": [
; JSON: "phase_launch_materialization": "device_stub"
; JSON: "phase_launch_reason": "one original kernel launch with reusable parameters and unchanged stream"
; JSON: "phase_launch_stream": "explicit"
; JSON: "phase_launch_supported": true
; JSON: ],
; JSON: "phase_launch_materialization": "device_stub"
; JSON: "phase_launch_reason": "one original kernel launch with reusable parameters and unchanged stream"
; JSON: "phase_launch_stream": "explicit"
; JSON: "phase_launch_supported": true
