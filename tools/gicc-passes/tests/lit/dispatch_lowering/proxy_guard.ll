; REQUIRES: gicc_lowering
; CPU_PROXY_ENQUEUE requires runtime support: when GICC_PROXY_ENABLED
; is not set in the environment, GICCDispatchLowering must emit a
; report_fatal_error mentioning the env var (and exit non-zero) instead
; of silently producing IR that would link against runtime helpers
; that aren't compiled in.
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/k_one_put_meta.json \
; RUN:    %t.metadir/_Z9k_one_putPN4gicc9DeviceCtxEim.json
; RUN: not env -u GICC_PROXY_ENABLED \
; RUN:     GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_cpu_proxy.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-dispatch-lowering' \
; RUN:          -disable-output %s 2>&1 | %FileCheck %s

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

; CHECK: GICC_PROXY_ENABLED is not set
