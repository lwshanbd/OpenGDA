; Verify the trace call lands immediately before the launch wrapper
; call at every launch site (not just the first), and that the trace
; function is reused across multiple sites of the same kernel.
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/k_one_put_meta.json \
; RUN:    %t.metadir/_Z9k_one_putPN4gicc9DeviceCtxEim.json
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-trace-synthesis' \
; RUN:          -S %s | %FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

@.str.gicc = private unnamed_addr constant [17 x i8] c"gicc.launch_site\00", section "llvm.metadata"
@.str.f    = private unnamed_addr constant [12 x i8] c"launch.hpp\00\00", section "llvm.metadata"

@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }]
  [{ ptr, ptr, ptr, i32, ptr }
   { ptr @_ZN4gicc6launchITnDaXadL_Z9k_one_putPN4gicc9DeviceCtxEimEEvRNS_7RuntimeEiy,
     ptr @.str.gicc,
     ptr @.str.f,
     i32 47,
     ptr null }],
   section "llvm.metadata"

define linkonce_odr void @_ZN4gicc6launchITnDaXadL_Z9k_one_putPN4gicc9DeviceCtxEimEEvRNS_7RuntimeEiy(
    ptr %rt, i32 %peer, i64 %size) {
  ret void
}

define void @two_calls(ptr %rt, i32 %p1, i32 %p2, i64 %s) {
  call void @_ZN4gicc6launchITnDaXadL_Z9k_one_putPN4gicc9DeviceCtxEimEEvRNS_7RuntimeEiy(ptr %rt, i32 %p1, i64 %s)
  call void @_ZN4gicc6launchITnDaXadL_Z9k_one_putPN4gicc9DeviceCtxEimEEvRNS_7RuntimeEiy(ptr %rt, i32 %p2, i64 %s)
  ret void
}

; CHECK-LABEL: define void @two_calls
; CHECK: call void @gicc_trace_k_one_put(ptr %rt, i32 %p1, i64 %s)
; CHECK-NEXT: call void @_ZN4gicc6launchITnDa{{.*}}(ptr %rt, i32 %p1, i64 %s)
; CHECK: call void @gicc_trace_k_one_put(ptr %rt, i32 %p2, i64 %s)
; CHECK-NEXT: call void @_ZN4gicc6launchITnDa{{.*}}(ptr %rt, i32 %p2, i64 %s)

; Trace function should be defined exactly once even though there are
; two launch sites.
; CHECK: define internal void @gicc_trace_k_one_put(
; CHECK-NOT: define internal void @gicc_trace_k_one_put(
