; At the early-simplification extension point, clang may not yet have inlined
; dim3's constructor. Verify that feature extraction follows the actual
; constructor -> ABI load chain without requiring source text. (The extractor
; also accepts the pre-SROA form with an intervening memcpy.)
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

%struct.dim3 = type { i32, i32, i32 }

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

declare void @_ZN4dim3C2Ejjj(ptr, i32, i32, i32)
define linkonce_odr void @_ZN4gicc6launchITnDaXadL_Z9k_const_szPN4gicc9DeviceCtxEEEvRNS_7RuntimeE(ptr %rt, i64 %grid.xy, i32 %grid.z, i64 %block.xy, i32 %block.z) {
  ret void
}

define void @main(ptr %rt) {
  %grid = alloca %struct.dim3, align 4
  %block = alloca %struct.dim3, align 4
  call void @_ZN4dim3C2Ejjj(ptr %grid, i32 8, i32 1, i32 1)
  call void @_ZN4dim3C2Ejjj(ptr %block, i32 1, i32 1, i32 1)
  %grid.xy = load i64, ptr %grid, align 4
  %grid.z.ptr = getelementptr inbounds i8, ptr %grid, i64 8
  %grid.z = load i32, ptr %grid.z.ptr, align 4
  %block.xy = load i64, ptr %block, align 4
  %block.z.ptr = getelementptr inbounds i8, ptr %block, i64 8
  %block.z = load i32, ptr %block.z.ptr, align 4
  call void @_ZN4gicc6launchITnDaXadL_Z9k_const_szPN4gicc9DeviceCtxEEEvRNS_7RuntimeE(ptr %rt, i64 %grid.xy, i32 %grid.z, i64 %block.xy, i32 %block.z)
  ret void
}

; JSON-DAG: "schema_version": 6
; JSON-DAG: "launch_grid": {
; JSON-DAG: "x": 8
; JSON-DAG: "y": 1
; JSON-DAG: "z": 1
; JSON-DAG: "launch_block": {
; JSON-DAG: "x": 1
; JSON-DAG: "grid_blocks": 8
; JSON-DAG: "threads_per_block": 1
