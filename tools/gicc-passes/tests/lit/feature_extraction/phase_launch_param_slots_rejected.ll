; A local parameter array is not enough: the cell type at every ABI slot must
; agree with device metadata.  Otherwise compiler-owned guards would read a
; formal using an unproved representation and phase launch stays disabled.
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
@_Z9k_const_szPN4gicc9DeviceCtxE = external constant ptr

@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }]
  [{ ptr, ptr, ptr, i32, ptr }
   { ptr @_ZN4gicc6launchITnDaXadL_Z9k_const_szPN4gicc9DeviceCtxEEEvRNS_7RuntimeE,
     ptr @.str.gicc, ptr @.str.f, i32 47, ptr null }],
  section "llvm.metadata"

declare i32 @hipLaunchKernel(ptr, i64, i32, i64, i32, ptr, i64, ptr)

define linkonce_odr void @_ZN4gicc6launchITnDaXadL_Z9k_const_szPN4gicc9DeviceCtxEEEvRNS_7RuntimeE(
    ptr %rt, i64 %grid.xy, i32 %grid.z, i64 %block.xy, i32 %block.z,
    i64 %shmem, ptr %stream) {
entry:
  ; Device formal zero is ptr, but the supplied cell is i64.
  %wrong.ctx.arg = alloca i64, align 8
  %params = alloca [1 x ptr], align 8
  store i64 0, ptr %wrong.ctx.arg, align 8
  store ptr %wrong.ctx.arg, ptr %params, align 8
  %rc = call i32 @hipLaunchKernel(
      ptr @_Z9k_const_szPN4gicc9DeviceCtxE,
      i64 %grid.xy, i32 %grid.z, i64 %block.xy, i32 %block.z,
      ptr %params, i64 %shmem, ptr %stream)
  ret void
}

define void @main(ptr %rt, ptr %stream) {
entry:
  call void @_ZN4gicc6launchITnDaXadL_Z9k_const_szPN4gicc9DeviceCtxEEEvRNS_7RuntimeE(
      ptr %rt, i64 1, i32 1, i64 1, i32 1, i64 0, ptr %stream)
  ret void
}

; JSON: "kernel_argument_slot_count": 1
; JSON: "kernel_argument_slot_reason": "kernel parameter slot type disagrees with device metadata"
; JSON: "kernel_argument_slots_exact": false
; JSON: "phase_launch_reason": "kernel parameter slot type disagrees with device metadata"
; JSON: "phase_launch_supported": false
