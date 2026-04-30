; Mimic the minimod halo_kernel shape:
;
;   __global__ void halo_kernel(DeviceCtx* ctx,
;                               int left_target, int right_target,
;                               int my_buf,
;                               long left_src_off, long left_dst_off,
;                               long right_src_off, long right_dst_off,
;                               long halo_bytes,
;                               int has_left, int has_right) {
;       if (has_left)  gicc::put_no_db(ctx, left_target, my_buf,
;                                      left_dst_off, my_buf,
;                                      left_src_off, halo_bytes);
;       if (has_right) gicc::put_no_db(ctx, right_target, my_buf,
;                                      right_dst_off, my_buf,
;                                      right_src_off, halo_bytes);
;       gicc::flush(ctx);
;   }
;
; Discovery (in feature-extract mode) writes
; ${GICC_META_DIR}/halo_kernel.json. We FileCheck the resulting JSON.
;
; RUN: rm -rf %t.metadir
; RUN: env GICC_MODE=feature-extract GICC_META_DIR=%t.metadir \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-device-discovery' \
; RUN:          -disable-output %s 2>&1 | %FileCheck %s --check-prefix=STDERR
; RUN: cat %t.metadir/halo_kernel.json | %FileCheck %s --check-prefix=JSON

target triple = "amdgcn-amd-amdhsa"

declare void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(ptr, i32, i32, i64, i32, i64, i64)
declare void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr)

define amdgpu_kernel void @halo_kernel(ptr %ctx,
                                       i32 %left_target,
                                       i32 %right_target,
                                       i32 %my_buf,
                                       i64 %left_src_off,
                                       i64 %left_dst_off,
                                       i64 %right_src_off,
                                       i64 %right_dst_off,
                                       i64 %halo_bytes,
                                       i32 %has_left,
                                       i32 %has_right) {
entry:
  %cl = icmp ne i32 %has_left, 0
  br i1 %cl, label %put_left, label %check_right

put_left:
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %left_target, i32 %my_buf, i64 %left_dst_off,
      i32 %my_buf, i64 %left_src_off, i64 %halo_bytes)
  br label %check_right

check_right:
  %cr = icmp ne i32 %has_right, 0
  br i1 %cr, label %put_right, label %do_flush

put_right:
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %right_target, i32 %my_buf, i64 %right_dst_off,
      i32 %my_buf, i64 %right_src_off, i64 %halo_bytes)
  br label %do_flush

do_flush:
  call void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr %ctx)
  ret void
}

; STDERR: [discovery] kernel halo_kernel has 3 sites
; STDERR: wrote {{.*}}halo_kernel.json

; JSON-DAG: "kernel_simple": "halo_kernel"
; JSON-DAG: "kind": "put_no_db"
; JSON-DAG: "kind": "param_truthy"
; JSON-DAG: "kind": "flush"
