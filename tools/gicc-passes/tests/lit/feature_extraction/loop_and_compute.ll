; Locks in the v2-schema additions: when the kernel JSON carries a
; loop descriptor + compute_before count, the feature record must
; surface in_loop=true, a structured `loop` sub-object, and a numeric
; `compute_before_flops` (not the null sentinel).
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/k_loop_with_compute_meta.json \
; RUN:    %t.metadir/_Z16k_loop_with_compPN4gicc9DeviceCtxEiiil.json
; RUN: env GICC_MODE=feature-extract GICC_META_DIR=%t.metadir \
; RUN:                              GICC_FEATURES_OUT=%t.features.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-feature-extraction' \
; RUN:          -disable-output %s 2>&1 | %FileCheck %s --check-prefix=STDERR
; RUN: cat %t.features.json | %FileCheck %s --check-prefix=JSON

target triple = "x86_64-unknown-linux-gnu"

@.str.gicc = private unnamed_addr constant [17 x i8] c"gicc.launch_site\00", section "llvm.metadata"
@.str.f    = private unnamed_addr constant [12 x i8] c"launch.hpp\00\00", section "llvm.metadata"

@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }]
  [{ ptr, ptr, ptr, i32, ptr }
   { ptr @_ZN4gicc6launchITnDaXadL_Z16k_loop_with_compPN4gicc9DeviceCtxEiiilEEEvRNS_7RuntimeE,
     ptr @.str.gicc,
     ptr @.str.f,
     i32 11,
     ptr null }],
   section "llvm.metadata"

define linkonce_odr void @_ZN4gicc6launchITnDaXadL_Z16k_loop_with_compPN4gicc9DeviceCtxEiiilEEEvRNS_7RuntimeE(ptr %rt) {
  ret void
}

define void @main(ptr %rt) {
  call void @_ZN4gicc6launchITnDaXadL_Z16k_loop_with_compPN4gicc9DeviceCtxEiiilEEEvRNS_7RuntimeE(ptr %rt)
  ret void
}

; STDERR: [feature-extract] wrote {{.*}}features.json

; JSON-DAG: "schema_version": 2
; JSON-DAG: "kernel": "k_loop_with_comp"
; JSON-DAG: "in_loop": true
; JSON-DAG: "iv_start": 0
; JSON-DAG: "iv_step": 1
; JSON-DAG: "bound_known": true
; JSON-DAG: "bound_param_idx": 3
; JSON-DAG: "compute_before_flops": 7
