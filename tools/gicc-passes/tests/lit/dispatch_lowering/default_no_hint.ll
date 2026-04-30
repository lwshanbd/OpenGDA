; Without GICC_HINT_IN set, DispatchLowering still runs with the safe
; default (DWQ_TRIGGER). The placeholder is replaced — no orphaned
; placeholder calls or declarations remain.
;
; RUN: env GICC_MODE=lower \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-dispatch-lowering' \
; RUN:          -S %s | %FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

declare void @gicc.runtime.put_no_db.placeholder(ptr, i32, i32, i64, i32, i64, i64)

define void @t(ptr %rt, i32 %peer, i64 %size) {
  call void @gicc.runtime.put_no_db.placeholder(
      ptr %rt, i32 %peer, i32 0, i64 0, i32 0, i64 0, i64 %size),
      !gicc.site_id !0
  ret void
}

!0 = !{!"site::0"}

; CHECK-LABEL: define void @t
; CHECK: call void @gicc_runtime_dwq_enqueue(ptr %rt, i32 %peer
; CHECK-NOT: gicc.runtime.put_no_db.placeholder
; CHECK-NOT: declare void @gicc.runtime.put_no_db.placeholder
