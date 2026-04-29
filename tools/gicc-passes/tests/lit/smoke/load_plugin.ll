; Verify the plugin loads via opt -load-pass-plugin and that the
; sentinel pass picks up GICC_MODE from the environment.
;
; RUN: env GICC_MODE=passthrough \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-sentinel' \
; RUN:          -S %s 2>&1 | %FileCheck %s

; CHECK: [gicc-pass] mode=passthrough

target triple = "x86_64-unknown-linux-gnu"

define void @main() {
  ret void
}
