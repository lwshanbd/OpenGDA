; Running the pass again changes nothing: instrumented functions and the
; mirror function are marked, and the mirror's own store is a repeat.
;
; RUN: env GICC_MODE=chunk-lower %opt -load-pass-plugin=%gicc_passes_so \
; RUN:   -passes=gicc-nodb-mirror,gicc-nodb-mirror -S %s | %FileCheck %s

target triple = "nvptx64-nvidia-cuda"

; CHECK-LABEL: define void @k(
; CHECK:       call ptr @__gicc_nodb_mirror(
; CHECK:       store float %x, ptr %{{.*}}, align 4, !gicc.repeat
; CHECK-NOT:   call ptr @__gicc_nodb_mirror
; CHECK:       ret void
; CHECK-NOT:   define internal ptr @__gicc_nodb_mirror.
define void @k(ptr %v, float %x) {
  store float %x, ptr %v, align 4
  ret void
}
