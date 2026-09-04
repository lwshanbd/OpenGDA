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
; RUN: grep -A2 '"legal_paths"' %t.features.json | \
; RUN:     %FileCheck %s --check-prefix=LEGAL

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

define void @main(ptr %rt, i64 %dynamic.grid.xy) {
  call void @_ZN4gicc6launchITnDaXadL_Z16k_loop_with_compPN4gicc9DeviceCtxEiiilEEEvRNS_7RuntimeE(ptr %rt, i64 %dynamic.grid.xy, i32 1, i64 4294967297, i32 1, i32 1, i32 2, i32 64, i64 4096)
  ret void
}

; STDERR: [feature-extract] wrote {{.*}}features.json

; JSON-DAG: "schema_version": 6
; JSON-DAG: "kernel": "k_loop_with_comp"
; JSON-DAG: "in_loop": true
; JSON-DAG: "iv_start": 0
; JSON-DAG: "iv_step": 1
; JSON-DAG: "bound_known": true
; JSON-DAG: "bound_param_idx": 3
; JSON-DAG: "bound_param_type": "i32"
; JSON-DAG: "compute_before_flops": 7
; Dynamic grid x/y remain unknown rather than guessed; the constant block is
; still recovered independently.
; JSON-DAG: "launch_grid": {
; JSON-DAG: "x": null
; JSON-DAG: "y": null
; JSON-DAG: "z": 1
; JSON-DAG: "grid_blocks": null
; JSON-DAG: "threads_per_block": 1
; Device metadata says size and loop bound are formals. Host LTO binds both
; to constants at this launch site without reading source.
; JSON-DAG: "size_kind": "param"
; JSON-DAG: "size_bytes": 4096
; JSON-DAG: "size_log2": 12
; JSON-DAG: "trip_count": 64
; JSON-DAG: "iter_estimate": 64
;
; The loop body issues the same descriptor every iteration: peer, buffers,
; offsets and size are all kernel formals or literals, none of them the
; induction variable. That is the case the host can stage once and trigger
; trip_count times, and it is invisible to a runtime looking at one call.
; JSON-DAG: "descriptor_reusable": true
; JSON-DAG: "buffer_reusable": true
; A modeled loop becomes a batched host placeholder. The current pass can
; materialize trigger or proxy for that shape, but not IPC/hybrid.
; LEGAL: "legal_paths": [
; LEGAL-NEXT: {{ *}}"proxy",
; LEGAL-NEXT: {{ *}}"trigger"
; LEGAL-NOT: ipc
; batch_size is carried in the kernel JSON rather than derived, because
; grouping transfers by completion point needs the CFG.
; JSON-DAG: "batch_size": 3
