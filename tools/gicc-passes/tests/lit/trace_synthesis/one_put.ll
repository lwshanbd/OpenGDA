; Single unguarded put_no_db — TraceSynthesis emits a trace function
; with one placeholder call carrying a gicc.site_id metadata operand
; and inserts the call before the launch wrapper.
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

define linkonce_odr void @_ZN4gicc6launchITnDaXadL_Z9k_one_putPN4gicc9DeviceCtxEimEEvRNS_7RuntimeEiy(ptr %rt, i32 %peer, i64 %size) {
  ret void
}

define void @main(ptr %rt, i32 %peer, i64 %size) {
  call void @_ZN4gicc6launchITnDaXadL_Z9k_one_putPN4gicc9DeviceCtxEimEEvRNS_7RuntimeEiy(
      ptr %rt, i32 %peer, i64 %size)
  ret void
}

; CHECK-LABEL: define void @main
; CHECK: call void @gicc_trace_k_one_put(ptr %rt, i32 %peer, i64 %size)
; CHECK-NEXT: call void @_ZN4gicc6launchITnDa
; CHECK-LABEL: define internal void @gicc_trace_k_one_put(ptr %rt, i32 %peer, i64 %size)
; CHECK: call void @gicc.runtime.put_no_db.placeholder(ptr %rt, i32 %peer, i32 0, i64 0, i32 0, i64 0, i64 %size){{.*}}!gicc.site_id ![[MD:[0-9]+]]
; CHECK: ![[MD]] = !{!"k.cpp:1:k_one_put::0"}
