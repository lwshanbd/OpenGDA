; Through the whole default pipeline, which fires both the EarlySimplification
; and the OptimizerLast extension points, lower mode must still put exactly
; one trace call before each launch. The OptimizerLast block used to run the
; shared trace pass a second time; that call also lost the fastcc the
; optimizer had given the internal trace function in between, and the
; mismatch made the call undefined behaviour.
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/k_one_put_meta.json \
; RUN:    %t.metadir/_Z9k_one_putPN4gicc9DeviceCtxEim.json
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='default<O2>' \
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

declare void @launch_kernel(ptr, i32, i64)

define linkonce_odr void @_ZN4gicc6launchITnDaXadL_Z9k_one_putPN4gicc9DeviceCtxEimEEvRNS_7RuntimeEiy(
    ptr %rt, i32 %peer, i64 %size) noinline {
  call void @launch_kernel(ptr %rt, i32 %peer, i64 %size)
  ret void
}

define void @two_calls(ptr %rt, i32 %p1, i32 %p2, i64 %s) {
  call void @_ZN4gicc6launchITnDaXadL_Z9k_one_putPN4gicc9DeviceCtxEimEEvRNS_7RuntimeEiy(ptr %rt, i32 %p1, i64 %s)
  call void @_ZN4gicc6launchITnDaXadL_Z9k_one_putPN4gicc9DeviceCtxEimEEvRNS_7RuntimeEiy(ptr %rt, i32 %p2, i64 %s)
  ret void
}

; The trace may be inlined (its first act is the IPC lookup) or called.
; CHECK-LABEL: define void @two_calls
; CHECK: {{@gicc_trace_k_one_put|@gicc_runtime_peer_ipc_base}}(ptr %rt, i32 %p1
; CHECK-NOT: {{@gicc_trace_k_one_put|@gicc_runtime_peer_ipc_base}}(
; CHECK: call void @_ZN4gicc6launchITnDa{{.*}}(ptr %rt, i32 %p1, i64 %s)
; CHECK: {{@gicc_trace_k_one_put|@gicc_runtime_peer_ipc_base}}(ptr %rt, i32 %p2
; CHECK-NOT: {{@gicc_trace_k_one_put|@gicc_runtime_peer_ipc_base}}(
; CHECK: call void @_ZN4gicc6launchITnDa{{.*}}(ptr %rt, i32 %p2, i64 %s)
; CHECK-NEXT: ret void
