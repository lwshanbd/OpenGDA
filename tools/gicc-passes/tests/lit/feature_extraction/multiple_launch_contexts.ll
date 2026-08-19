; One device operation may be reached from multiple host launch sites, but the
; current hint lowers that shared device op only once. Verify that extraction
; emits one record with two contexts instead of duplicate decision IDs, and
; that host constants are bound back to the size and loop-bound formals.
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/k_loop_with_compute_meta.json \
; RUN:    %t.metadir/_Z16k_loop_with_compPN4gicc9DeviceCtxEiiil.json
; RUN: env GICC_MODE=feature-extract GICC_META_DIR=%t.metadir \
; RUN:                              GICC_FEATURES_OUT=%t.features.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-feature-extraction' \
; RUN:          -disable-output %s
; RUN: cat %t.features.json | %FileCheck %s --check-prefix=JSON
; RUN: cat %t.features.json | %FileCheck %s --check-prefix=COUNT

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

define linkonce_odr void @_ZN4gicc6launchITnDaXadL_Z16k_loop_with_compPN4gicc9DeviceCtxEiiilEEEvRNS_7RuntimeE(ptr %rt, i64 %grid.xy, i32 %grid.z, i64 %block.xy, i32 %block.z, i32 %peer, i32 %buf, i32 %n, i64 %bytes) {
  ret void
}

define void @main(ptr %rt) {
  call void @_ZN4gicc6launchITnDaXadL_Z16k_loop_with_compPN4gicc9DeviceCtxEiiilEEEvRNS_7RuntimeE(ptr %rt, i64 4294967297, i32 1, i64 4294967297, i32 1, i32 1, i32 2, i32 64, i64 4096)
  call void @_ZN4gicc6launchITnDaXadL_Z16k_loop_with_compPN4gicc9DeviceCtxEiiilEEEvRNS_7RuntimeE(ptr %rt, i64 4294967304, i32 1, i64 4294967297, i32 1, i32 1, i32 2, i32 32, i64 65536)
  ret void
}

; The top-level decision facts are null because the two contexts differ.
; JSON: "grid_blocks": null
; Both proven contexts remain available to the decider.
; JSON: "launch_contexts": [
; JSON: "grid_blocks": 1
; JSON: "size_bytes": 4096
; JSON: "static_callsite_count": 1
; JSON: "trip_count": 64
; JSON: "grid_blocks": 8
; JSON: "size_bytes": 65536
; JSON: "static_callsite_count": 1
; JSON: "trip_count": 32
; JSON: "schema_version": 6
; JSON: "site_id": "k.cpp:5:k_loop_with_comp::0"
; JSON: "size_bytes": null
; JSON: "size_log2": null
; JSON: "static_launch_sites": 2
; JSON: "trip_count": null
; COUNT-COUNT-1: "site_id": "k.cpp:5:k_loop_with_comp::0"
