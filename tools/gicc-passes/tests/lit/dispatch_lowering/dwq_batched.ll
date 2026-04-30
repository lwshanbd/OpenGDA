; DWQ_BATCHED dispatch: two consecutive same-BB placeholders collapse
; into a single gicc_runtime_dwq_enqueue_batched call. Six stack arrays
; (peers / dst_bufs / dst_offs / src_bufs / src_offs / sizes) are
; allocated at function entry; both placeholders' args are stored
; into slots [0] and [1]; one batched call submits all 6×N values.
;
; RUN: env GICC_MODE=lower GICC_HINT_IN=%S/../Inputs/hint_dwq_batched.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-dispatch-lowering' \
; RUN:          -S %s | %FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

declare void @gicc.runtime.put_no_db.placeholder(ptr, i32, i32, i64, i32, i64, i64)

define void @t(ptr %rt, i32 %p1, i64 %s1, i32 %p2, i64 %s2) {
entry:
  call void @gicc.runtime.put_no_db.placeholder(
      ptr %rt, i32 %p1, i32 0, i64 0, i32 0, i64 0, i64 %s1),
      !gicc.site_id !0
  call void @gicc.runtime.put_no_db.placeholder(
      ptr %rt, i32 %p2, i32 0, i64 0, i32 0, i64 0, i64 %s2),
      !gicc.site_id !1
  ret void
}

!0 = !{!"site::0"}
!1 = !{!"site::1"}

; Six stack arrays at entry, sized for n=2.
; CHECK-LABEL: define void @t
; CHECK: alloca i32, i32 2
; CHECK: alloca i32, i32 2
; CHECK: alloca i64, i32 2
; CHECK: alloca i32, i32 2
; CHECK: alloca i64, i32 2
; CHECK: alloca i64, i32 2

; Stores into slot[0] via GEP at index 0.
; CHECK: getelementptr i32, ptr %dwq.peers, i32 0
; CHECK: store i32 %p1, ptr %{{.*}}, align 4
; CHECK: getelementptr i64, ptr %dwq.sizes, i32 0
; CHECK: store i64 %s1

; Stores into slot[1] via GEP at index 1.
; CHECK: getelementptr i32, ptr %dwq.peers, i32 1
; CHECK: store i32 %p2
; CHECK: getelementptr i64, ptr %dwq.sizes, i32 1
; CHECK: store i64 %s2

; One batched call.
; CHECK: call void @gicc_runtime_dwq_enqueue_batched(ptr %rt, i32 2, ptr %dwq.peers, ptr %dwq.dst_bufs, ptr %dwq.dst_offs, ptr %dwq.src_bufs, ptr %dwq.src_offs, ptr %dwq.sizes)
; CHECK-NOT: gicc.runtime.put_no_db.placeholder
