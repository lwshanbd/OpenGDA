; A single put_no_db inside a host-knowable loop:
;
;   for (int i = 0; i < n; i++)
;       gicc::put_no_db(ctx, peer, buf, i*bytes_per, buf, i*bytes_per,
;                              bytes_per);
;
; The OpTemplate JSON in Inputs/k_loop_meta.json marks the op as
; loop-wrapped (iv_param=3 = `n`, iv_start=0, iv_step=1) with dst_off
; and src_off referencing ArgRef::LoopIv. TraceSynthesis must emit a
; host-side loop around the placeholder call: a phi i64 in loop.head,
; an icmp slt against the bound, the placeholder call inside the body,
; and an `add i64 ..., 1` increment.
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/k_loop_meta.json \
; RUN:    %t.metadir/_Z6k_loopPN4gicc9DeviceCtxEiiil.json
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-trace-synthesis' \
; RUN:          -S %s | %FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

@.str.gicc = private unnamed_addr constant [17 x i8] c"gicc.launch_site\00", section "llvm.metadata"
@.str.f    = private unnamed_addr constant [12 x i8] c"launch.hpp\00\00", section "llvm.metadata"

@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }]
  [{ ptr, ptr, ptr, i32, ptr }
   { ptr @_ZN4gicc6launchITnDaXadL_Z6k_loopPN4gicc9DeviceCtxEiiilEEvRNS_7RuntimeEiiiy,
     ptr @.str.gicc,
     ptr @.str.f,
     i32 47,
     ptr null }],
   section "llvm.metadata"

define linkonce_odr void @_ZN4gicc6launchITnDaXadL_Z6k_loopPN4gicc9DeviceCtxEiiilEEvRNS_7RuntimeEiiiy(
    ptr %rt, i32 %peer, i32 %buf, i32 %n, i64 %bytes_per) {
  ret void
}

define void @main(ptr %rt, i32 %peer, i32 %buf, i32 %n, i64 %bytes_per) {
  call void @_ZN4gicc6launchITnDaXadL_Z6k_loopPN4gicc9DeviceCtxEiiilEEvRNS_7RuntimeEiiiy(
      ptr %rt, i32 %peer, i32 %buf, i32 %n, i64 %bytes_per)
  ret void
}

; The launch site receives a trace call before the original launch wrapper.
; CHECK-LABEL: define void @main
; CHECK: call void @gicc_trace_k_loop(ptr %rt, i32 %peer, i32 %buf, i32 %n, i64 %bytes_per)
; CHECK-NEXT: call void @_ZN4gicc6launchI

; The synthesized trace function wraps the placeholder in a host-side
; loop. We expect: a loop.head BB with a phi i64 starting at 0; an
; icmp slt comparing against the (sext i32→i64) bound; a placeholder
; call inside loop.body using the iv as part of dst_off; and an
; `add i64 ..., 1` step in the body.

; The block names contain `:` so LLVM round-trips them as quoted strings.
; CHECK-LABEL: define internal void @gicc_trace_k_loop(ptr %rt, i32 %peer, i32 %buf, i32 %n, i64 %bytes_per)
; CHECK: "loop.head.k.cpp:5:k_loop::0":
; CHECK: %iv = phi i64 [ 0, %{{.*}} ], [ %iv.next, %"loop.body.k.cpp:5:k_loop::0" ]
; CHECK: %cmp = icmp slt i64 %iv, %{{.*}}
; CHECK: br i1 %cmp, label %"loop.body.k.cpp:5:k_loop::0", label %"loop.exit.k.cpp:5:k_loop::0"
; CHECK: "loop.body.k.cpp:5:k_loop::0":
; CHECK: call void @gicc.runtime.put_no_db.placeholder(ptr %rt, i32 %peer, i32 %buf, i64 %{{.*}}, i32 %buf, i64 %{{.*}}, i64 %bytes_per){{.*}}!gicc.site_id
; CHECK: %iv.next = add i64 %iv, 1
; CHECK: br label %"loop.head.k.cpp:5:k_loop::0"
; CHECK: "loop.exit.k.cpp:5:k_loop::0":
