; CPU_PROXY_ENQUEUE lowering: the host-side placeholder call is erased
; (the actual RDMA work runs device-side and is serviced by the CPU
; proxy thread), and the kernel's per-kernel JSON is updated with
; `"proxy_aware": true` so device-lowering (Task 8) knows to preserve
; the device-side body for this kernel.
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/k_one_put_meta.json \
; RUN:    %t.metadir/_Z9k_one_putPN4gicc9DeviceCtxEim.json
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_cpu_proxy.json \
; RUN:     GICC_PROXY_ENABLED=1 \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-dispatch-lowering' \
; RUN:          -S %s | %FileCheck %s --check-prefix=IR
; RUN: cat %t.metadir/_Z9k_one_putPN4gicc9DeviceCtxEim.json \
; RUN:     | %FileCheck %s --check-prefix=JSON

target triple = "x86_64-unknown-linux-gnu"

@.str.gicc = private unnamed_addr constant [17 x i8] c"gicc.launch_site\00", section "llvm.metadata"
@.str.f    = private unnamed_addr constant [12 x i8] c"launch.hpp\00\00", section "llvm.metadata"

; Launch wrapper for gicc::launch<&k_one_put> — required so
; collectLaunchInventory can locate the matching kernel JSON and feed
; the (site_id -> hk_capable, kernel_mangled) map used by the
; dispatch-lowering cross-check + proxy_aware write-back.
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }]
  [{ ptr, ptr, ptr, i32, ptr }
   { ptr @_ZN4gicc6launchITnDaXadL_Z9k_one_putPN4gicc9DeviceCtxEimEEvRNS_7RuntimeEiy,
     ptr @.str.gicc,
     ptr @.str.f,
     i32 47,
     ptr null }],
   section "llvm.metadata"

declare void @gicc.runtime.put_no_db.placeholder(ptr, i32, i32, i64, i32, i64, i64)

define linkonce_odr void @_ZN4gicc6launchITnDaXadL_Z9k_one_putPN4gicc9DeviceCtxEimEEvRNS_7RuntimeEiy(ptr %rt, i32 %peer, i64 %size) {
  ret void
}

define void @gicc_trace_k_one_put(ptr %rt, i32 %peer, i64 %size) {
  call void @gicc.runtime.put_no_db.placeholder(
      ptr %rt, i32 %peer, i32 0, i64 0, i32 0, i64 0, i64 %size),
      !gicc.site_id !0
  ret void
}

define void @main(ptr %rt, i32 %peer, i64 %size) {
  call void @_ZN4gicc6launchITnDaXadL_Z9k_one_putPN4gicc9DeviceCtxEimEEvRNS_7RuntimeEiy(
      ptr %rt, i32 %peer, i64 %size)
  ret void
}

!0 = !{!"k.cpp:1:k_one_put::0"}

; The placeholder call must be erased; no IPC / DWQ runtime calls
; should be emitted in its place.
; IR-LABEL: define void @gicc_trace_k_one_put
; IR-NOT:   gicc.runtime.put_no_db.placeholder
; IR-NOT:   call ptr @gicc_runtime_peer_ipc_base
; IR-NOT:   call void @gicc_runtime_dwq_enqueue
; IR-NOT:   call i32 @hipMemcpyAsync

; The kernel JSON must record proxy_aware=true so device-lowering
; (Task 8) knows to preserve the device-side put_no_db body.
; JSON: "proxy_aware": true
