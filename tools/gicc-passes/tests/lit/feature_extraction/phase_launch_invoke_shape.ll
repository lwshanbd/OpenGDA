; Verify that an exception-aware caller edge does not mask a schedule whose
; materialization point is the independently audited HIP launch in the
; annotated wrapper. The caller invoke and both successors remain untouched.
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
declare i32 @__gxx_personality_v0(...)

define linkonce_odr void @_ZN4gicc6launchITnDaXadL_Z9k_const_szPN4gicc9DeviceCtxEEEvRNS_7RuntimeE(
    ptr %rt, i64 %grid.xy, i32 %grid.z, i64 %block.xy, i32 %block.z,
    i64 %shmem, ptr %stream) {
entry:
  %ctx.arg = alloca ptr, align 8
  %params = alloca [1 x ptr], align 8
  store ptr null, ptr %ctx.arg, align 8
  store ptr %ctx.arg, ptr %params, align 8
  %rc = call i32 @hipLaunchKernel(
      ptr @_Z9k_const_szPN4gicc9DeviceCtxE,
      i64 %grid.xy, i32 %grid.z, i64 %block.xy, i32 %block.z,
      ptr %params, i64 %shmem, ptr %stream)
  ret void
}

define void @main(ptr %rt, ptr %stream)
    personality ptr @__gxx_personality_v0 {
entry:
  invoke void @_ZN4gicc6launchITnDaXadL_Z9k_const_szPN4gicc9DeviceCtxEEEvRNS_7RuntimeE(
      ptr %rt, i64 4294967304, i32 1, i64 4294967297, i32 1,
      i64 0, ptr %stream)
      to label %done unwind label %cleanup

done:
  ret void

cleanup:
  %exception = landingpad { ptr, i32 }
      cleanup
  resume { ptr, i32 } %exception
}

; JSON: "launch_contexts": [
; JSON: "kernel_argument_slot_count": 1
; JSON: "kernel_argument_slots_exact": true
; JSON: "phase_launch_materialization": "wrapper"
; JSON: "phase_launch_reason": "one original kernel launch with reusable parameters and unchanged stream"
; JSON: "phase_launch_stream": "explicit"
; JSON: "phase_launch_supported": true
; JSON: ],
; JSON: "phase_launch_materialization": "wrapper"
; JSON: "phase_launch_supported": true
