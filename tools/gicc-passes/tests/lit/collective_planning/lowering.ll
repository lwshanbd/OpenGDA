; REQUIRES: gicc_lowering
; The LTO pass accepts only content-addressed catalog selections, and lowers a
; compiler-owned message policy to control flow. A tampered threshold fails
; closed and preserves the semantic anchor call.
;
; RUN: env GICC_MODE=lower \
; RUN:   GICC_COLLECTIVE_HINT_IN=%S/../Inputs/collective_hint_uniform.json \
; RUN:   %opt -load-pass-plugin=%gicc_passes_so \
; RUN:        -passes='gicc-collective-planning' -S %s \
; RUN:   | %FileCheck %s --check-prefix=UNIFORM
; RUN: env GICC_MODE=lower \
; RUN:   GICC_COLLECTIVE_HINT_IN=%S/../Inputs/collective_hint_policy.json \
; RUN:   %opt -load-pass-plugin=%gicc_passes_so \
; RUN:        -passes='gicc-collective-planning' -S %s \
; RUN:   | %FileCheck %s --check-prefix=POLICY
; RUN: env GICC_MODE=lower \
; RUN:   GICC_COLLECTIVE_HINT_IN=%S/../Inputs/collective_hint_tampered.json \
; RUN:   %opt -load-pass-plugin=%gicc_passes_so \
; RUN:        -passes='gicc-collective-planning' -S %s \
; RUN:   | %FileCheck %s --check-prefix=REJECT

target triple = "x86_64-unknown-linux-gnu"

@.anchor = private unnamed_addr constant [173 x i8] c"gicc.collective.anchor.v1;family=allreduce_sum_f32_inplace;contract=registered_v1;algorithm=baseline_auto;count_arg=0;ppn_arg=1;element_bytes=4;thresholds_bytes=4096,262144\00"
@.tree = private unnamed_addr constant [175 x i8] c"gicc.collective.candidate.v1;family=allreduce_sum_f32_inplace;contract=registered_v1;algorithm=double_tree;communication_graph=tree;step_complexity=O(log_ranks);topology=flat\00"
@.ring = private unnamed_addr constant [177 x i8] c"gicc.collective.candidate.v1;family=allreduce_sum_f32_inplace;contract=registered_v1;algorithm=hier_ring;communication_graph=ring;step_complexity=O(nodes);topology=hierarchical\00"
@.file = private unnamed_addr constant [12 x i8] c"redacted.cc\00"

@llvm.global.annotations = appending global [3 x { ptr, ptr, ptr, i32, ptr }] [
  { ptr, ptr, ptr, i32, ptr } { ptr @anchor, ptr @.anchor, ptr @.file, i32 1, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @double_tree, ptr @.tree, ptr @.file, i32 2, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @hier_ring, ptr @.ring, ptr @.file, i32 3, ptr null }
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

define void @caller(i64 %dynamic_count) {
  %scaled = mul i64 %dynamic_count, 2
  call void @anchor(i64 %scaled, i32 8)
  ret void
}

; UNIFORM-LABEL: define void @caller
; UNIFORM: call void @double_tree(i64 %scaled, i32 8)
; UNIFORM-NOT: call void @anchor

; POLICY-LABEL: define void @caller
; POLICY: %gicc.collective.bytes = mul i64 %scaled, 4
; POLICY: icmp ule i64 %gicc.collective.bytes, 4096
; POLICY: call void @double_tree(i64 %scaled, i32 8)
; POLICY: icmp ule i64 %gicc.collective.bytes, 262144
; POLICY: call void @hier_ring(i64 %scaled, i32 8)
; POLICY: call void @anchor(i64 %scaled, i32 8)

; REJECT-LABEL: define void @caller
; REJECT: call void @anchor(i64 %scaled, i32 8)
; REJECT-NOT: call void @double_tree
; REJECT-NOT: call void @hier_ring
