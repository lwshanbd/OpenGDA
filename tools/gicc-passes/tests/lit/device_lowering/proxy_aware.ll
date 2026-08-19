; REQUIRES: gicc_lowering
; GICCDeviceLowering: hint-aware put_no_db preservation. The pass reads
; `proxy_aware: bool` from the per-kernel JSON written by Task 2's
; dispatch-lowering. When the kernel has any CPU_PROXY_ENQUEUE site
; (proxy_aware=true), the device-side put_no_db body MUST be preserved
; so the device pushes a TransferCmd into the proxy ring at runtime.
; When proxy_aware=false (or no JSON), the host trace owns the work and
; the device-side body is erased.
;
; The preserved body is additionally GATED to a single grid-wide lead
; thread (workitem.id.x==0 && workgroup.id.x==0), mirroring the flush
; lowering — each gicc::put site is one logical transfer, so the device
; push must happen once, not once per thread (else the proxy ring
; overflows and rt.reset()'s drain hangs).
; This fixture assigns the preserved op block_slot=9 and follows it with a
; quiet whose metadata intentionally has no block_slot (the intervening flush
; would own the host/DWQ batch in a real kernel).  The pass must obtain
; gridDim.x from the real dispatch-packet intrinsic; declarations with
; invented llvm.amdgcn.* names survive opt and fail only at device link time.
; Both the issuing block and its ring lane must use 9 % gridDim.x, and quiet
; must conservatively drain every launched lane that the proxy site can use.
;
; Two kernels in one module exercise both branches:
;   kernel_dwq_only      → proxy_aware=false → put_no_db erased.
;   kernel_proxy_aware   → proxy_aware=true  → put_no_db preserved + guarded.
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/proxy_aware_meta/kernel_dwq_only.json    %t.metadir/
; RUN: cp %S/../Inputs/proxy_aware_meta/kernel_proxy_aware.json %t.metadir/
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-device-lowering' \
; RUN:          -S %s | %FileCheck %s

target triple = "amdgcn-amd-amdhsa"

declare void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimmi(ptr, i32, i32, i64, i32, i64, i64, i32)
declare void @_ZN4gicc5quietEPN4gicc9DeviceCtxEii(ptr, i32, i32)

define amdgpu_kernel void @kernel_dwq_only(ptr %ctx, i32 %p, i32 %b, i64 %off, i64 %sz) {
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimmi(
      ptr %ctx, i32 %p, i32 %b, i64 %off, i32 %b, i64 %off, i64 %sz, i32 0)
  ret void
}

define amdgpu_kernel void @kernel_proxy_aware(ptr %ctx, i32 %p, i32 %b, i64 %off, i64 %sz) {
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimmi(
      ptr %ctx, i32 %p, i32 %b, i64 %off, i32 %b, i64 %off, i64 %sz, i32 0)
  call void @_ZN4gicc5quietEPN4gicc9DeviceCtxEii(ptr %ctx, i32 0, i32 3)
  ret void
}

; CHECK-LABEL: define amdgpu_kernel void @kernel_dwq_only
; CHECK-NOT:   call void @_ZN4gicc9put_no_db
; CHECK:       ret void

; CHECK-LABEL: define amdgpu_kernel void @kernel_proxy_aware
; CHECK:       call i32 @llvm.amdgcn.workitem.id.x()
; CHECK:       call i32 @llvm.amdgcn.workgroup.id.x()
; CHECK:       call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
; CHECK:       getelementptr i8, ptr addrspace(4) {{.*}}, i64 4
; CHECK:       getelementptr i8, ptr addrspace(4) {{.*}}, i64 12
; CHECK-NOT:   @llvm.amdgcn.grid.size.x
; CHECK-NOT:   @llvm.amdgcn.workgroup.size.x
; CHECK:       [[LANE:%.*]] = urem i32 9, {{%.*}}
; CHECK:       br i1
; CHECK:       call void @_ZN4gicc9put_no_db{{.*}}i32 [[LANE]])
; CHECK:       icmp ult i32 {{%.*}}, 10
; CHECK:       br i1
; CHECK:       call void @_ZN4gicc5quietEPN4gicc9DeviceCtxEii({{.*}}i32 {{%.*}}, i32 0)
; CHECK:       ret void
