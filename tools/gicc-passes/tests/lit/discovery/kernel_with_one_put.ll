; Verify GICCDeviceDiscovery enumerates GICC call sites in a kernel and
; reports each site_id + op kind to stderr.
;
; RUN: env GICC_MODE=feature-extract \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-device-discovery' \
; RUN:          -disable-output %s 2>&1 | %FileCheck %s

target triple = "amdgcn-amd-amdhsa"

declare void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(ptr, i32, i32, i64, i32, i64, i64)
declare void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr)

define amdgpu_kernel void @halo_kernel(ptr %ctx, i32 %peer, i32 %buf,
                                       i64 %dst, i32 %src_buf,
                                       i64 %src, i64 %size) {
entry:
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %peer, i32 %buf, i64 %dst,
      i32 %src_buf, i64 %src, i64 %size)
  call void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr %ctx)
  ret void
}

; CHECK: [discovery] kernel halo_kernel has 2 sites
; CHECK-DAG: [discovery]   site_id={{.*}}halo_kernel::0 kind=put_no_db
; CHECK-DAG: [discovery]   site_id={{.*}}halo_kernel::1 kind=flush
