; Task 1 (CPU Proxy plan) — HK softening.
;
; A non-host-knowable argument (here: the result of
; llvm.amdgcn.workitem.id.x feeding the 'size' arg) used to make the HK
; pass emit `error:` and was effectively a hard build failure. Now it
; should:
;   1. Print a `warning:` mentioning `CPU_PROXY_ENQUEUE` (no error,
;      pass exit code is success).
;   2. Set the per-site `hk_capable` bit to false in the on-disk
;      kernel JSON written by GICCDeviceDiscovery.
;
; Pipeline runs Discovery (which writes the initial JSON) then HK
; (which reruns the analysis and rewrites the JSON with the capability
; bit). Mode=feature-extract so Discovery actually persists JSON.
;
; RUN: rm -rf %t.metadir
; RUN: env GICC_MODE=feature-extract GICC_META_DIR=%t.metadir \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-device-discovery,gicc-hk-analysis' \
; RUN:          -disable-output %s 2>&1 | %FileCheck %s --check-prefix=STDERR
; RUN: cat %t.metadir/bad_kernel.json | %FileCheck %s --check-prefix=JSON

target triple = "amdgcn-amd-amdhsa"

declare i32  @llvm.amdgcn.workitem.id.x()
declare void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(ptr, i32, i32, i64, i32, i64, i64)

define amdgpu_kernel void @bad_kernel(ptr %ctx, i32 %peer) {
entry:
  %tid32 = call i32 @llvm.amdgcn.workitem.id.x()
  %tid64 = zext i32 %tid32 to i64
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %peer, i32 0, i64 0, i32 0, i64 0, i64 %tid64)
  ret void
}

; HK pass must surface a warning (NOT an error) and the warning text
; must mention CPU_PROXY_ENQUEUE so users know how the site will be
; routed.
; STDERR-NOT: error: gicc::
; STDERR: warning: gicc::put_no_db: argument 'size' is not host-knowable
; STDERR-SAME: CPU_PROXY_ENQUEUE
; STDERR: note: depends on llvm.amdgcn.workitem.id.x

; The on-disk kernel template must record the capability bit so the
; host-side passes (feature extraction, dispatch lowering) can route
; the site to CPU_PROXY_ENQUEUE without re-running HK.
; JSON: "hk_capable": false
; JSON: "hk_fail_reason"
