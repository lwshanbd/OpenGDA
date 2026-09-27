; An alloca after a checked store in the entry block stays static: the
; entry block's allocas move to its top before any split.
;
; RUN: env GICC_MODE=chunk-lower %opt -load-pass-plugin=%gicc_passes_so \
; RUN:   -passes=gicc-nodb-mirror -S %s | %FileCheck %s

target triple = "nvptx64-nvidia-cuda"

declare void @use(ptr addrspace(5))

; CHECK-LABEL: define void @k(
; CHECK-NEXT:  %a = alloca i32
; CHECK-NEXT:  %b = alloca i32
; CHECK:       call ptr @__gicc_nodb_mirror(
define void @k(ptr %v, float %x) {
  %a = alloca i32, align 4, addrspace(5)
  store float %x, ptr %v, align 4
  %b = alloca i32, align 4, addrspace(5)
  call void @use(ptr addrspace(5) %a)
  call void @use(ptr addrspace(5) %b)
  ret void
}
