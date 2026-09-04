; The scalar repeated-descriptor placeholder has an exact one-to-one runtime
; ABI. Dispatch lowering may rename it only when the bound hint requests the
; compiler-owned REUSE_LOOP_DESCRIPTOR transform on DWQ_TRIGGER.
;
; RUN: env GICC_MODE=lower \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_reuse_loop_descriptor.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-dispatch-lowering' -S %s | %FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

declare void @gicc.runtime.put_no_db.repeated.placeholder(
    ptr, i32, i32, i32, i64, i32, i64, i64)

define void @t(ptr %rt, i32 %n, i32 %peer, i32 %buf, i64 %size) {
  call void @gicc.runtime.put_no_db.repeated.placeholder(
      ptr %rt, i32 %n, i32 %peer, i32 %buf, i64 0,
      i32 %buf, i64 0, i64 %size), !gicc.site_id !0
  ret void
}

!0 = !{!"reuse.cpp:5:k_reuse::0"}

; CHECK-LABEL: define void @t
; CHECK: call void @gicc_runtime_dwq_enqueue_repeated(ptr %rt, i32 %n, i32 %peer, i32 %buf, i64 0, i32 %buf, i64 0, i64 %size)
; CHECK-NOT: gicc.runtime.put_no_db.repeated.placeholder
