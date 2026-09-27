; Nothing happens outside chunk-lower, or on the host.
;
; RUN: env GICC_MODE=passthrough %opt -load-pass-plugin=%gicc_passes_so \
; RUN:   -passes=gicc-nodb-mirror -S %s | %FileCheck %s
; RUN: sed 's/nvptx64-nvidia-cuda/aarch64-unknown-linux-gnu/' %s | \
; RUN:   env GICC_MODE=chunk-lower %opt -load-pass-plugin=%gicc_passes_so \
; RUN:   -passes=gicc-nodb-mirror -S | %FileCheck %s

target triple = "nvptx64-nvidia-cuda"

; CHECK-NOT: ompx__nodb
; CHECK-NOT: __gicc_nodb_mirror
define void @k(ptr %v, float %x) {
  store float %x, ptr %v, align 4
  ret void
}
