; FeatureExtraction emits one record per launch site × op. A put_no_db
; with a constant 4096-byte size becomes
;   "size_kind": "const",
;   "size_log2": 12
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/k_const_size_meta.json \
; RUN:    %t.metadir/_Z9k_const_szPN4gicc9DeviceCtxE.json
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
   { ptr @_ZN4gicc6launchITnDaXadL_Z9k_const_szPN4gicc9DeviceCtxEEEvRNS_7RuntimeE,
     ptr @.str.gicc,
     ptr @.str.f,
     i32 47,
     ptr null }],
   section "llvm.metadata"

define linkonce_odr void @_ZN4gicc6launchITnDaXadL_Z9k_const_szPN4gicc9DeviceCtxEEEvRNS_7RuntimeE(ptr %rt) {
  ret void
}

define void @main(ptr %rt) {
  call void @_ZN4gicc6launchITnDaXadL_Z9k_const_szPN4gicc9DeviceCtxEEEvRNS_7RuntimeE(ptr %rt)
  ret void
}

; STDERR: [feature-extract] wrote {{.*}}features.json

; JSON-DAG: "kernel": "k_const_sz"
; JSON-DAG: "op_kind": "put_no_db"
; JSON-DAG: "size_kind": "const"
; JSON-DAG: "size_log2": 12
; JSON-DAG: "peer_kind": "const"
