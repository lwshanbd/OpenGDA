; The collective inventory contains source-free call/context facts and an
; ABI-filtered semantic catalog. It must not expose function/source names.
;
; RUN: env GICC_MODE=feature-extract GICC_COLLECTIVE_OUT=%t.json \
; RUN:   %opt -load-pass-plugin=%gicc_passes_so \
; RUN:        -passes='gicc-collective-planning' -disable-output %s
; RUN: %FileCheck %s --check-prefix=JSON < %t.json

target triple = "x86_64-unknown-linux-gnu"

@.anchor = private unnamed_addr constant [173 x i8] c"gicc.collective.anchor.v1;family=allreduce_sum_f32_inplace;contract=registered_v1;algorithm=baseline_auto;count_arg=0;ppn_arg=1;element_bytes=4;thresholds_bytes=4096,262144\00"
@.tree = private unnamed_addr constant [175 x i8] c"gicc.collective.candidate.v1;family=allreduce_sum_f32_inplace;contract=registered_v1;algorithm=double_tree;communication_graph=tree;step_complexity=O(log_ranks);topology=flat\00"
@.ring = private unnamed_addr constant [177 x i8] c"gicc.collective.candidate.v1;family=allreduce_sum_f32_inplace;contract=registered_v1;algorithm=hier_ring;communication_graph=ring;step_complexity=O(nodes);topology=hierarchical\00"
@.bad = private unnamed_addr constant [169 x i8] c"gicc.collective.candidate.v1;family=allreduce_sum_f32_inplace;contract=registered_v1;algorithm=wrong_abi;communication_graph=ring;step_complexity=O(nodes);topology=flat\00"
@.file = private unnamed_addr constant [12 x i8] c"redacted.cc\00"

@llvm.global.annotations = appending global [4 x { ptr, ptr, ptr, i32, ptr }] [
  { ptr, ptr, ptr, i32, ptr } { ptr @anchor, ptr @.anchor, ptr @.file, i32 1, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @double_tree, ptr @.tree, ptr @.file, i32 2, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @hier_ring, ptr @.ring, ptr @.file, i32 3, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @wrong_abi, ptr @.bad, ptr @.file, i32 4, ptr null }
], section "llvm.metadata"

define void @anchor(i64 %count, i32 %ppn) noinline {
  ret void
}

define void @double_tree(i64 %count, i32 %ppn) noinline {
  ret void
}

define void @hier_ring(i64 %count, i32 %ppn) noinline {
  ret void
}

define i32 @wrong_abi(i64 %count, i32 %ppn) noinline {
  ret i32 0
}

define void @caller(i64 %dynamic_count) {
  %scaled = mul i64 %dynamic_count, 2
  call void @anchor(i64 %scaled, i32 8)
  %after = add i64 %scaled, 7
  ret void
}

; JSON-DAG: "schema_version": "gicc-collective-inventory-v1"
; JSON-DAG: "compiler_only": true
; JSON-DAG: "source_visible": false
; JSON-DAG: "algorithm": "baseline_auto"
; JSON-DAG: "arithmetic_before": 1
; JSON-DAG: "arithmetic_after": 1
; JSON-DAG: "element_bytes": 4
; JSON-DAG: "message_bytes": null
; JSON-DAG: "ranks_per_node": {
; JSON-DAG: "kind": "constant"
; JSON-DAG: "value": 8
; JSON-DAG: "thresholds_bytes": "4096,262144"
; JSON-DAG: "algorithm": "double_tree"
; JSON-DAG: "algorithm": "hier_ring"
; JSON-NOT: "algorithm": "wrong_abi"
; JSON-NOT: redacted.cc
; JSON-NOT: double_tree(
