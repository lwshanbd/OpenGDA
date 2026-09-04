; Device discovery records a conservative group-level legality proof.  The
; first kernel may move its shared trigger across arithmetic only; the second
; is rejected because a store could update a registered source buffer.
;
; RUN: rm -rf %t.metadir
; RUN: env GICC_MODE=feature-extract GICC_META_DIR=%t.metadir \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-device-discovery' -disable-output %s
; RUN: cat %t.metadir/group_safe.json | \
; RUN:     %FileCheck %s --check-prefix=SAFE
; RUN: cat %t.metadir/group_write.json | \
; RUN:     %FileCheck %s --check-prefix=WRITE

target triple = "amdgcn-amd-amdhsa"

declare void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
    ptr, i32, i32, i64, i32, i64, i64)
declare void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr)
declare i32 @_ZN24__hip_builtin_blockIdx_t7__get_xEv() nounwind readnone
declare i32 @_ZN24__hip_builtin_blockDim_t7__get_xEv() nounwind readnone
declare i32 @_ZN25__hip_builtin_threadIdx_t7__get_xEv() nounwind readnone

@unknown_state = addrspace(1) global i32 0

; Deliberately carries no memory attributes. Producer-frontier discovery must
; inspect the available body and prove that it has no externally visible
; write, as is required for HIP's pre-inlining builtin wrappers.
define i32 @read_index_wrapper() {
entry:
  ret i32 7
}

define amdgpu_kernel void @group_safe(
    ptr %ctx, i32 %top, i32 %bottom, i32 %buf) {
entry:
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %top, i32 %buf, i64 0,
      i32 %buf, i64 0, i64 4096)
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %bottom, i32 %buf, i64 4096,
      i32 %buf, i64 4096, i64 4096)
  %compute = fadd float 1.000000e+00, 2.000000e+00
  call void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr %ctx)
  ret void
}

define amdgpu_kernel void @group_write(
    ptr %ctx, ptr addrspace(1) %out, ptr addrspace(1) %sum,
    i32 %top, i32 %bottom, i32 %buf, i64 %limit) {
entry:
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %top, i32 %buf, i64 0,
      i32 %buf, i64 0, i64 4096)
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %bottom, i32 %buf, i64 4096,
      i32 %buf, i64 4096, i64 4096)
  %block = call i32 @_ZN24__hip_builtin_blockIdx_t7__get_xEv()
  %width = call i32 @_ZN24__hip_builtin_blockDim_t7__get_xEv()
  %thread = call i32 @_ZN25__hip_builtin_threadIdx_t7__get_xEv()
  %block_base = mul i32 %block, %width
  %global_i32 = add i32 %block_base, %thread
  %element = zext i32 %global_i32 to i64
  %in_bounds = icmp ult i64 %element, %limit
  br i1 %in_bounds, label %produce, label %complete

produce:
  %slot = getelementptr i32, ptr addrspace(1) %out, i64 %element
  store volatile i32 1, ptr addrspace(1) %slot
  %old = atomicrmw fadd ptr addrspace(1) %sum, float 1.000000e+00 monotonic
  br label %complete

complete:
  call void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr %ctx)
  ret void
}

define amdgpu_kernel void @group_unknown(
    ptr %ctx, i32 %top, i32 %bottom, i32 %buf) {
entry:
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %top, i32 %buf, i64 0,
      i32 %buf, i64 0, i64 4096)
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %bottom, i32 %buf, i64 4096,
      i32 %buf, i64 4096, i64 4096)
  store volatile i32 1, ptr addrspace(1) @unknown_state
  call void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr %ctx)
  ret void
}

define amdgpu_kernel void @group_multi_buffer(
    ptr %ctx, ptr addrspace(1) %out,
    i32 %top, i32 %bottom, i32 %buf0, i32 %buf1) {
entry:
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %top, i32 %buf0, i64 0,
      i32 %buf0, i64 0, i64 4096)
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %bottom, i32 %buf1, i64 4096,
      i32 %buf1, i64 4096, i64 4096)
  %opaque_index = call i32 @read_index_wrapper()
  %opaque_slot = getelementptr i32, ptr addrspace(1) %out, i32 %opaque_index
  store volatile i32 1, ptr addrspace(1) %opaque_slot
  call void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr %ctx)
  ret void
}

; SAFE: "completion_site_id": "?:?:group_safe::2"
; SAFE: "group_early_trigger_legal": true
; SAFE: "group_early_trigger_reason": "proved mandatory flush and no intervening memory writes"
; SAFE: "producer_frontier": {
; SAFE-DAG: "ordinary_store_sites": 0
; SAFE-DAG: "unknown_write_sites": 0
; SAFE-DAG: "write_footprint_known": false
; SAFE: "site_id": "?:?:group_safe::0"
; SAFE: "completion_site_id": "?:?:group_safe::2"
; SAFE: "group_early_trigger_legal": true
; SAFE: "site_id": "?:?:group_safe::1"

; WRITE: "completion_site_id": "?:?:group_write::2"
; WRITE: "group_early_trigger_legal": false
; WRITE: "group_early_trigger_reason": "intervening instruction may write a registered source buffer"
; WRITE: "producer_frontier": {
; WRITE-DAG: "ordinary_store_params": [
; WRITE-NEXT: 1
; WRITE-DAG: "atomic_write_params": [
; WRITE-NEXT: 2
; WRITE-DAG: "atomic_write_sites": 1
; WRITE-DAG: "ordinary_store_sites": 1
; WRITE-DAG: "unknown_write_sites": 0
; WRITE-DAG: "write_footprint_known": true
; WRITE-DAG: "buffer_identity_guardable": true
; WRITE-DAG: "producer_pointer_param": 1
; WRITE-DAG: "source_buffer_index_param": 5
; WRITE-DAG: "buffer_identity_guard_reason": "one producer pointer and one shared i32 source-buffer formal can be guarded at launch"
; WRITE-DAG: "producer_domains_known": true
; WRITE-DAG: "producer_store_domains": [
; WRITE-DAG: "address_exact": true
; WRITE-DAG: "byte_offset": {
; WRITE-DAG: "kind": "binop"
; WRITE-DAG: "op": "mul"
; WRITE-DAG: "name": "block_id_x"
; WRITE-DAG: "name": "block_size_x"
; WRITE-DAG: "name": "thread_id_x"
; WRITE-DAG: "value": 4
; WRITE-DAG: "byte_size": 4
; WRITE-DAG: "domain_exact": true
; WRITE-DAG: "pointer_param": 1
; WRITE-DAG: "predicates": [
; WRITE-DAG: "kind": "compare"
; WRITE-DAG: "op": "ult"
; WRITE-DAG: "param": 6
; WRITE-DAG: "required_value": true
; WRITE-DAG: "predicates_exact": true
; WRITE-DAG: "reason": "formal-rooted writes and exact local store domains recovered; transfer matching and side-effect partition remain unproved"

; RUN: cat %t.metadir/group_unknown.json | \
; RUN:     %FileCheck %s --check-prefix=UNKNOWN
; UNKNOWN: "producer_frontier": {
; UNKNOWN-DAG: "ordinary_store_sites": 0
; UNKNOWN-DAG: "unknown_write_sites": 1
; UNKNOWN-DAG: "write_footprint_known": false
; UNKNOWN-DAG: "buffer_identity_guardable": false
; UNKNOWN-DAG: "producer_domains_known": false
; UNKNOWN-DAG: "producer_store_domains": []
; UNKNOWN-DAG: "reason": "an intervening write is not rooted in a kernel pointer formal"

; RUN: cat %t.metadir/group_multi_buffer.json | \
; RUN:     %FileCheck %s --check-prefix=MULTI
; MULTI: "producer_frontier": {
; MULTI-DAG: "ordinary_store_params": [
; MULTI-NEXT: 1
; MULTI-DAG: "buffer_identity_guardable": false
; MULTI-DAG: "producer_pointer_param": null
; MULTI-DAG: "source_buffer_index_param": null
; MULTI-DAG: "buffer_identity_guard_reason": "requires one producer pointer and one shared i32 source-buffer formal"
; MULTI-DAG: "producer_domains_known": false
; MULTI-DAG: "address_exact": false
; MULTI-DAG: "byte_offset": {
; MULTI-DAG: "kind": "unknown"
; MULTI-DAG: "domain_exact": false
