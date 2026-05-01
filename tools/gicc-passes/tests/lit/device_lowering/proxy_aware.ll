; GICCDeviceLowering: hint-aware put_no_db preservation. The pass reads
; `proxy_aware: bool` from the per-kernel JSON written by Task 2's
; dispatch-lowering. When the kernel has any CPU_PROXY_ENQUEUE site
; (proxy_aware=true), the device-side put_no_db body MUST be preserved
; so the device pushes a TransferCmd into the proxy ring at runtime.
; When proxy_aware=false (or no JSON), the host trace owns the work and
; the device-side body is erased.
;
; Two kernels in one module exercise both branches:
;   kernel_dwq_only      → proxy_aware=false → put_no_db erased.
;   kernel_proxy_aware   → proxy_aware=true  → put_no_db preserved.
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/proxy_aware_meta/kernel_dwq_only.json    %t.metadir/
; RUN: cp %S/../Inputs/proxy_aware_meta/kernel_proxy_aware.json %t.metadir/
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-device-lowering' \
; RUN:          -S %s | %FileCheck %s

target triple = "amdgcn-amd-amdhsa"

declare void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(ptr, i32, i32, i64, i32, i64, i64)

define amdgpu_kernel void @kernel_dwq_only(ptr %ctx, i32 %p, i32 %b, i64 %off, i64 %sz) {
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %p, i32 %b, i64 %off, i32 %b, i64 %off, i64 %sz)
  ret void
}

define amdgpu_kernel void @kernel_proxy_aware(ptr %ctx, i32 %p, i32 %b, i64 %off, i64 %sz) {
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %p, i32 %b, i64 %off, i32 %b, i64 %off, i64 %sz)
  ret void
}

; CHECK-LABEL: define amdgpu_kernel void @kernel_dwq_only
; CHECK-NOT:   call void @_ZN4gicc9put_no_db
; CHECK:       ret void

; CHECK-LABEL: define amdgpu_kernel void @kernel_proxy_aware
; CHECK:       call void @_ZN4gicc9put_no_db
; CHECK:       ret void
