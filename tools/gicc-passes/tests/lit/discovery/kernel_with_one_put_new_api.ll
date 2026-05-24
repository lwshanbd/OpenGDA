; Phase 2.1 sibling of kernel_with_one_put.ll. Exercises classifyGICCCall
; on the NEW API names (gicc::put + gicc::quiet with a trailing `int lane`
; arg) — mangled to "3putE" / "5quietE" instead of "9put_no_db" / "5quietE".
; The discovery pass should classify them as PutNoDb / Quiet just like the
; legacy names, and the lane arg should be silently dropped by the trace
; template builder (see TraceTemplateBuilder::fillArgs).
;
; RUN: env GICC_MODE=feature-extract \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-device-discovery' \
; RUN:          -disable-output %s 2>&1 | %FileCheck %s

target triple = "amdgcn-amd-amdhsa"

; Mangled gicc::put(DeviceCtx*, int peer, int dst_buf, size_t dst_off,
;                   int src_buf, size_t src_off, size_t size, int lane)
declare void @_ZN4gicc3putEPN4gicc9DeviceCtxEiimimmi(
    ptr, i32, i32, i64, i32, i64, i64, i32)

; Mangled gicc::quiet(DeviceCtx*, int lane)
declare void @_ZN4gicc5quietEPN4gicc9DeviceCtxEi(ptr, i32)

define amdgpu_kernel void @halo_kernel(ptr %ctx, i32 %peer, i32 %buf,
                                       i64 %dst, i32 %src_buf,
                                       i64 %src, i64 %size, i32 %lane) {
entry:
  call void @_ZN4gicc3putEPN4gicc9DeviceCtxEiimimmi(
      ptr %ctx, i32 %peer, i32 %buf, i64 %dst,
      i32 %src_buf, i64 %src, i64 %size, i32 %lane)
  call void @_ZN4gicc5quietEPN4gicc9DeviceCtxEi(ptr %ctx, i32 %lane)
  ret void
}

; CHECK: [discovery] kernel halo_kernel has 2 sites
; CHECK-DAG: [discovery]   site_id={{.*}}halo_kernel::0 kind=put_no_db
; CHECK-DAG: [discovery]   site_id={{.*}}halo_kernel::1 kind=quiet
