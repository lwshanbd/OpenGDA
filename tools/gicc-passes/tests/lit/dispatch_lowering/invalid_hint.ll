; REQUIRES: gicc_lowering
; An untrusted decision file with an unknown dispatch must be rejected as a
; whole.  It must not silently map Unknown to DWQ_TRIGGER.  The compiler's
; fail-closed no-hint behavior is the hybrid IPC_OR_DWQ path.
;
; RUN: env GICC_MODE=lower \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_unknown_dispatch.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-dispatch-lowering' -S %s 2>%t.err \
; RUN:          | %FileCheck %s
; RUN: cat %t.err | %FileCheck %s --check-prefix=WARN

target triple = "x86_64-unknown-linux-gnu"

declare void @gicc.runtime.put_no_db.placeholder(ptr, i32, i32, i64, i32, i64, i64)

define void @trace(ptr %rt, i32 %peer, i32 %buf, i64 %off, i64 %size) {
entry:
  call void @gicc.runtime.put_no_db.placeholder(
      ptr %rt, i32 %peer, i32 %buf, i64 %off,
      i32 %buf, i64 %off, i64 %size), !gicc.site_id !0
  ret void
}

; CHECK-LABEL: define void @trace
; CHECK: call ptr @gicc_runtime_peer_ipc_base
; CHECK: br i1 {{.*}}, label %ipc, label %dwq
; CHECK: call void @gicc_runtime_dwq_enqueue
; CHECK-NOT: gicc.runtime.put_no_db.placeholder
; WARN: falling back to default IPC_OR_DWQ

!0 = !{!"unit.cpp:1:k::0"}
