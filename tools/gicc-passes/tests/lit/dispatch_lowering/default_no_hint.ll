; Without GICC_HINT_IN set, DispatchLowering applies the new default
; IPC_OR_DWQ hybrid lowering: a runtime branch on the peer's IPC base
; pointer chooses between hipMemcpyAsync (same-node IPC) and
; gicc_runtime_dwq_enqueue (off-node DWQ). The placeholder must be
; removed.
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
; CHECK: %[[BASE:.*]] = call ptr @gicc_runtime_peer_ipc_base(ptr %rt, i32 %peer, i32 0)
; CHECK: %[[HAS:.*]] = icmp ne ptr %[[BASE]], null
; CHECK: br i1 %[[HAS]], label %{{.*}}, label %{{.*}}
; CHECK: call i32 @hipMemcpyAsync
; CHECK: call void @gicc_runtime_dwq_enqueue(ptr %rt, i32 %peer
; CHECK-NOT: gicc.runtime.put_no_db.placeholder
; CHECK-NOT: declare void @gicc.runtime.put_no_db.placeholder
