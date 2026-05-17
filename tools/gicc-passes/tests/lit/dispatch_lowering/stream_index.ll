; REQUIRES: gicc_lowering
; IPC_PUSH with non-default stream_index=3: verify that the decider-supplied
; stream_index flows through to gicc_runtime_ipc_stream_indexed as a literal
; i32 3, not the default i32 0.
;
; RUN: env GICC_MODE=lower GICC_HINT_IN=%S/../Inputs/hint_ipc_push_stream3.json \
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
; CHECK: call ptr @gicc_runtime_peer_ipc_base(ptr %rt, i32 %peer, i32 0)
; CHECK: getelementptr i8, ptr %{{.*}}, i64 0
; CHECK: call ptr @gicc_runtime_local_buf_base(ptr %rt, i32 0)
; CHECK: getelementptr i8, ptr %{{.*}}, i64 0
; CHECK: call ptr @gicc_runtime_ipc_stream_indexed(ptr %rt, i32 3)
; CHECK: call i32 @hipMemcpyAsync(ptr %{{.*}}, ptr %{{.*}}, i64 %size, i32 3, ptr %{{.*}})
; CHECK-NOT: gicc.runtime.put_no_db.placeholder
; CHECK-NOT: gicc_runtime_ipc_stream_indexed(ptr {{.*}}, i32 0)
