; REQUIRES: gicc_lowering
; Direction-3 path: GICCDeviceLowering decides whether to preserve the
; device-side put_no_db body by reading hint.json directly via
; DispatchDecision::kernelHasProxySite. NO kernel JSON is supplied
; (META_DIR is empty), proving the device pass no longer depends on
; the host-pass-written `proxy_aware` bit.
;
; Two kernels exercise both branches under the same hint.json:
;   kernel_dwq_only      → site routed via defaultDispatch=DWQ_TRIGGER
;                          → device body erased
;   kernel_proxy_aware   → site explicitly routed to CPU_PROXY_ENQUEUE
;                          → device body preserved
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_kernel_proxy_aware.json \
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
