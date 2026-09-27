; Stores that cannot reach heap memory are left alone: a stack slot, a
; global variable, shared memory. So is a pipelined put's copy of a store
; (!gicc.repeat): the store it repeats is the one checked. A function with nothing to check gets no
; loads of the range either.
;
; RUN: env GICC_MODE=chunk-lower %opt -load-pass-plugin=%gicc_passes_so \
; RUN:   -passes=gicc-nodb-mirror -S %s | %FileCheck %s

target triple = "nvptx64-nvidia-cuda"

@g = global i32 0
@sh = addrspace(3) global [32 x i32] undef

define void @k(i32 %x, ptr %p) {
; CHECK-LABEL: define void @k(
; CHECK-NOT:   ompx__nodb
; CHECK-NOT:   __gicc_nodb_mirror
; CHECK:       ret void
  %a = alloca i32, align 4, addrspace(5)
  store i32 %x, ptr addrspace(5) %a, align 4
  store i32 %x, ptr @g, align 4
  %s = getelementptr [32 x i32], ptr addrspace(3) @sh, i64 0, i32 %x
  store i32 %x, ptr addrspace(3) %s, align 4
  store i32 %x, ptr %p, align 4, !gicc.repeat !0
  ret void
}

!0 = !{}
