; Two exact PUT intervals and one exact producer store use the same kernel
; formal namespace.  The compiler can therefore define the producer subset
; by checked byte overlap and the remainder by its logical complement; no
; source-level relation between the runtime scalar values is needed.
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/k_overlap_partition_meta.json \
; RUN:    %t.metadir/_Z11k_partition.json
; RUN: env GICC_MODE=feature-extract GICC_META_DIR=%t.metadir \
; RUN:                              GICC_FEATURES_OUT=%t.features.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-feature-extraction' \
; RUN:          -disable-output %s
; RUN: cat %t.features.json | %FileCheck %s --check-prefix=JSON
;
; A second convergent operation reachable under the opposite value of the
; same formal leaves no value that disables every non-duplicable operation.
; RUN: rm -rf %t.mismatch.metadir && mkdir -p %t.mismatch.metadir
; RUN: python3 %S/../Inputs/add_phase_guard_mismatch.py \
; RUN:   %S/../Inputs/k_overlap_partition_meta.json \
; RUN:   %t.mismatch.metadir/_Z11k_partition.json
; RUN: env GICC_MODE=feature-extract \
; RUN:     GICC_META_DIR=%t.mismatch.metadir \
; RUN:     GICC_FEATURES_OUT=%t.mismatch.features.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-feature-extraction' \
; RUN:          -disable-output %s
; RUN: cat %t.mismatch.features.json | \
; RUN:     %FileCheck %s --check-prefix=MISMATCH

target triple = "x86_64-unknown-linux-gnu"

@.str.gicc = private unnamed_addr constant [17 x i8] c"gicc.launch_site\00", section "llvm.metadata"
@.str.f = private unnamed_addr constant [12 x i8] c"launch.hpp\00\00", section "llvm.metadata"

@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }]
  [{ ptr, ptr, ptr, i32, ptr }
   { ptr @_ZN4gicc6launchIXadL_Z11k_partitionEEEv,
     ptr @.str.gicc, ptr @.str.f, i32 47, ptr null }],
  section "llvm.metadata"

define void @_ZN4gicc6launchIXadL_Z11k_partitionEEEv(
    ptr %rt, i64 %grid.xy, i32 %grid.z, i64 %block.xy, i32 %block.z,
    ptr %out, i32 %buf, i64 %top, i64 %bottom, i64 %size, i64 %stride,
    i1 %calculate_norm) {
entry:
  ret void
}

define void @main(ptr %rt, ptr %out, i32 %buf,
                  i64 %top, i64 %bottom, i64 %size, i64 %stride,
                  i1 %calculate_norm) {
entry:
  call void @_ZN4gicc6launchIXadL_Z11k_partitionEEEv(
      ptr %rt, i64 1, i32 1, i64 1, i32 1,
      ptr %out, i32 %buf, i64 %top, i64 %bottom,
      i64 %size, i64 %stride, i1 %calculate_norm)
  ret void
}

; JSON: "overlap_partition": {
; JSON-DAG: "boundary_predicate": "store_interval_overlaps_any_transfer_interval"
; JSON-DAG: "checked_interval_ends_required": true
; JSON-DAG: "exact": true
; JSON-DAG: "formal_binding": "same_kernel_formal_indices"
; JSON-DAG: "full_compute_region_partition_proved": true
; JSON-DAG: "mode": "checked_store_interval_overlap"
; JSON-DAG: "proof_scope": "ordinary_producer_store_instances"
; JSON-DAG: "remainder_predicate": "logical_complement_of_boundary"
; JSON-DAG: "side_effect_partition_proved": false
; JSON-DAG: "side_effect_safety_exact": true
; JSON-DAG: "side_effect_safety_mode": "all_nonduplicable_operations_disabled_by_formal_guard"
; JSON-DAG: "side_effect_free_guard": {
; JSON-DAG: "kind": "param_eq"
; JSON-DAG: "param": 7
; JSON-DAG: "value": false
; JSON-DAG: "side_effects_excluded_on_optimized_path": true
; JSON-DAG: "phase_sensitive_domains_known": true
; JSON-DAG: "phase_sensitive_sites": 1
; JSON-DAG: "producer_phase_sensitive_domains": [
; JSON-DAG: "operation": "_Z11__shfl_downfji"
; JSON-DAG: "guard_predicates_exact": true
; JSON-DAG: "store_instance_partition_complete": true
; JSON-DAG: "store_instance_partition_disjoint": true
; JSON-DAG: "site_id": "unit.cpp:10:k_partition::0"
; JSON-DAG: "site_id": "unit.cpp:11:k_partition::1"
; JSON: "checked_interval_guard_materialization"
; JSON: "device_phase_partition_materialization"
; JSON: "side_effect_guarded_fallback_materialization"
; JSON-NOT: "exact_transfer_intervals"
; JSON-NOT: "exact_producer_domains"

; MISMATCH: "overlap_partition": {
; MISMATCH-DAG: "exact": true
; MISMATCH-DAG: "side_effect_free_guard": null
; MISMATCH-DAG: "side_effect_safety_exact": false
; MISMATCH-DAG: "side_effect_safety_reason": "non-duplicable operations lack one shared direct i1-formal disabling predicate"
; MISMATCH: "side_effect_partition"
