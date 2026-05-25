; REQUIRES: gicc_lowering
; Phase B batched form: trace synth emits a single
; gicc.runtime.put_no_db.batched.placeholder per loop op (after the
; loop body has staged args into stack arrays). Dispatch lowering
; should rename the callee to gicc_runtime_dwq_enqueue_batched —
; signatures already match 1:1, no IR transformation needed.
;
; RUN: env GICC_MODE=lower GICC_HINT_IN=%S/../Inputs/hint_dwq_batched.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-dispatch-lowering' \
; RUN:          -S %s | %FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

declare void @gicc.runtime.put_no_db.batched.placeholder(
    ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr)

define void @t(ptr %rt, i32 %n, ptr %peers, ptr %dbs, ptr %dos,
                ptr %sbs, ptr %sos, ptr %szs) {
  call void @gicc.runtime.put_no_db.batched.placeholder(
      ptr %rt, i32 %n,
      ptr %peers, ptr %dbs, ptr %dos,
      ptr %sbs, ptr %sos, ptr %szs),
      !gicc.site_id !0
  ret void
}

!0 = !{!"site::0"}

; CHECK-LABEL: define void @t
; CHECK: call void @gicc_runtime_dwq_enqueue_batched(ptr %rt, i32 %n, ptr %peers, ptr %dbs, ptr %dos, ptr %sbs, ptr %sos, ptr %szs)
; CHECK-NOT: gicc.runtime.put_no_db.batched.placeholder
