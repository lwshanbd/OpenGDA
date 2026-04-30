; The 'size' argument is the result of llvm.amdgcn.workitem.id.x — non-HK.
; HK pass must report a source-line-precise error.
;
; RUN: env GICC_MODE=feature-extract \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-hk-analysis' \
; RUN:          -disable-output %s 2>&1 | %FileCheck %s

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

; CHECK: error: gicc::put_no_db: argument 'size' is not host-knowable
; CHECK: note: depends on llvm.amdgcn.workitem.id.x
