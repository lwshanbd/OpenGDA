; How many transfers reach the wire together?
;
;   __global__ void k_batched(DeviceCtx* ctx, int target, int dst_buf,
;                             int src_buf) {
;       for (int i = 0; i < 8; i++)
;           gicc::put_no_db(ctx, target, dst_buf, 0, src_buf, 0, 4096);
;       gicc::flush(ctx);                       // <- closes the group
;       gicc::put_no_db(ctx, target, dst_buf, 0, src_buf, 0, 4096);
;   }
;
; The loop's single call site runs 8 times and every one of those is
; released by the same flush, so its batch_size is 8 -- not 1, which is
; all a runtime observing one call could ever conclude. The trailing put
; belongs to a second group that no flush closes; the host's reset()
; drains it, so it stands alone at 1.
;
; The descriptor is also loop-invariant here (peer, buffers, offsets and
; size are all formals or literals, none of them the induction variable),
; which is what lets the host stage once and trigger 8 times.
;
; RUN: rm -rf %t.metadir
; RUN: env GICC_MODE=feature-extract GICC_META_DIR=%t.metadir \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-device-discovery' \
; RUN:          -disable-output %s 2>&1 | %FileCheck %s --check-prefix=STDERR
; RUN: cat %t.metadir/_Z9k_batchedPN4gicc9DeviceCtxEiii.json | \
; RUN:     %FileCheck %s --check-prefix=JSON
;
; Propagation into the decider's features.json is covered by
; feature_extraction/loop_and_compute.ll -- that pass runs on the HOST
; module and needs launch-site annotations, which a device module has no
; business carrying.

target triple = "amdgcn-amd-amdhsa"

declare void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(ptr, i32, i32, i64, i32, i64, i64)
declare void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr)

define amdgpu_kernel void @_Z9k_batchedPN4gicc9DeviceCtxEiii(
    ptr %ctx, i32 %target, i32 %dst_buf, i32 %src_buf) {
entry:
  br label %loop.header

loop.header:
  %iv  = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %cmp = icmp slt i32 %iv, 8
  br i1 %cmp, label %loop.body, label %after

loop.body:
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %target, i32 %dst_buf, i64 0,
      i32 %src_buf, i64 0, i64 4096)
  br label %loop.latch

loop.latch:
  %iv.next = add nsw i32 %iv, 1
  br label %loop.header

after:
  call void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr %ctx)
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %target, i32 %dst_buf, i64 0,
      i32 %src_buf, i64 0, i64 4096)
  ret void
}

; STDERR: [discovery] kernel k_batched has 3 sites

; Keys are printed alphabetically, so batch_size leads each op object.
; The looped put is site 0 and carries the whole group; the trailing put
; is site 2 and stands alone.
; JSON: "batch_size": 8
; JSON: "site_id": "{{.*}}k_batched{{.*}}::0"
; JSON: "batch_size": 1
; JSON: "site_id": "{{.*}}k_batched{{.*}}::2"
