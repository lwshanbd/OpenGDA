; IPC_OR_DWQ lowering: replace placeholder with a runtime branch on
; %peer_base != null. Same-node peers (mapped IPC base) take the
; hipMemcpyAsync path; off-node peers fall through to the DWQ path.
; Both paths converge on the original successor, and the original
; placeholder call is gone.
;
; RUN: env GICC_MODE=lower GICC_HINT_IN=%S/../Inputs/hint_ipc_or_dwq.json \
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

!0 = !{!"k.cpp:1:k_one_put::0"}

; CHECK-LABEL: define void @t
; CHECK: %[[BASE:.*]] = call ptr @gicc_runtime_peer_ipc_base(ptr %rt, i32 %peer, i32 0)
; CHECK: %[[HAS:.*]] = icmp ne ptr %[[BASE]], null
; CHECK: br i1 %[[HAS]], label %[[IPC:.*]], label %[[DWQ:.*]]
;
; CHECK: [[IPC]]:
; CHECK: getelementptr i8, ptr %[[BASE]], i64 0
; CHECK: call ptr @gicc_runtime_local_buf_base(ptr %rt, i32 0)
; CHECK: call ptr @gicc_runtime_ipc_stream(ptr %rt)
; CHECK: call i32 @hipMemcpyAsync(ptr %{{.*}}, ptr %{{.*}}, i64 %size, i32 3, ptr %{{.*}})
; CHECK: br label %[[DONE:.*]]
;
; CHECK: [[DWQ]]:
; CHECK: call void @gicc_runtime_dwq_enqueue(ptr %rt, i32 %peer, i32 0, i64 0, i32 0, i64 0, i64 %size)
; CHECK: br label %[[DONE]]
;
; CHECK: [[DONE]]:
; CHECK: ret void
;
; CHECK-NOT: gicc.runtime.put_no_db.placeholder
