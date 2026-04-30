; Mimic a kernel that performs N puts in a host-knowable for-loop:
;
;   __global__ void put_kernel(DeviceCtx* ctx, int target, int dst_buf,
;                              int src_buf, int n, long bytes_per) {
;       for (int i = 0; i < n; i++)
;           gicc::put_no_db(ctx, target, dst_buf, i*bytes_per,
;                                 src_buf, i*bytes_per, bytes_per);
;   }
;
; Discovery (in feature-extract mode) writes
; ${GICC_META_DIR}/_Z10put_kernelPN4gicc9DeviceCtxEiiiil.json including
; a "loop" object (iv_param=4 = `n`) and at least one ArgRef with
; kind=loop_iv inside the dst_off / src_off children.
;
; RUN: rm -rf %t.metadir
; RUN: env GICC_MODE=feature-extract GICC_META_DIR=%t.metadir \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-device-discovery' \
; RUN:          -disable-output %s 2>&1 | %FileCheck %s --check-prefix=STDERR
; RUN: cat %t.metadir/_Z10put_kernelPN4gicc9DeviceCtxEiiiil.json | \
; RUN:     %FileCheck %s --check-prefix=JSON

target triple = "amdgcn-amd-amdhsa"

declare void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(ptr, i32, i32, i64, i32, i64, i64)

define amdgpu_kernel void @_Z10put_kernelPN4gicc9DeviceCtxEiiiil(
    ptr %ctx, i32 %target, i32 %dst_buf, i32 %src_buf,
    i32 %n, i64 %bytes_per) {
entry:
  br label %loop.preheader

loop.preheader:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %loop.preheader ], [ %iv.next, %loop.latch ]
  %cmp = icmp slt i32 %iv, %n
  br i1 %cmp, label %loop.body, label %exit

loop.body:
  %iv64    = sext i32 %iv to i64
  %off     = mul i64 %iv64, %bytes_per
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %target, i32 %dst_buf, i64 %off,
      i32 %src_buf, i64 %off, i64 %bytes_per)
  br label %loop.latch

loop.latch:
  %iv.next = add nsw i32 %iv, 1
  br label %loop.header

exit:
  ret void
}

; STDERR: [discovery] kernel put_kernel has 1 sites
; STDERR: wrote {{.*}}_Z10put_kernelPN4gicc9DeviceCtxEiiiil.json

; The JSON is pretty-printed with alphabetically sorted keys. The "args"
; object lands before the "loop" object; inside args the dst_off ArgRef
; encodes  mul ( sext (loop_iv) , bytes_per ).
;
; JSON: "args":
; JSON: "dst_off":
; JSON: "kind": "binop"
; JSON: "kind": "loop_iv"
; JSON: "loop":
; JSON: "in_loop": true
; JSON: "iv_param": 4
; JSON: "iv_start": 0
; JSON: "iv_step": 1
