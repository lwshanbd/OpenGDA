; Two guarded put_no_db's mimicking the minimod halo_kernel pattern.
; The trace function has two icmp+br pairs, one per guard (param_truthy
; on has_left and has_right).
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/k_two_puts_guarded_meta.json \
; RUN:    %t.metadir/_Z18k_two_puts_guardedPN4gicc9DeviceCtxEiimii.json
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-trace-synthesis' \
; RUN:          -S %s | %FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

@.str.gicc = private unnamed_addr constant [17 x i8] c"gicc.launch_site\00", section "llvm.metadata"
@.str.f    = private unnamed_addr constant [12 x i8] c"launch.hpp\00\00", section "llvm.metadata"

@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }]
  [{ ptr, ptr, ptr, i32, ptr }
   { ptr @_ZN4gicc6launchITnDaXadL_Z18k_two_puts_guardedPN4gicc9DeviceCtxEiimiiEEvRNS_7RuntimeEiimii,
     ptr @.str.gicc,
     ptr @.str.f,
     i32 47,
     ptr null }],
   section "llvm.metadata"

define linkonce_odr void @_ZN4gicc6launchITnDaXadL_Z18k_two_puts_guardedPN4gicc9DeviceCtxEiimiiEEvRNS_7RuntimeEiimii(
    ptr %rt, i32 %left, i32 %right, i64 %size, i32 %hl, i32 %hr) {
  ret void
}

define void @main(ptr %rt, i32 %left, i32 %right, i64 %size, i32 %hl, i32 %hr) {
  call void @_ZN4gicc6launchITnDaXadL_Z18k_two_puts_guardedPN4gicc9DeviceCtxEiimiiEEvRNS_7RuntimeEiimii(
      ptr %rt, i32 %left, i32 %right, i64 %size, i32 %hl, i32 %hr)
  ret void
}

; CHECK-LABEL: define internal void @gicc_trace_k_two_puts_guarded
; CHECK: icmp ne i32 %has_left, 0
; CHECK: br i1
; CHECK: call void @gicc.runtime.put_no_db.placeholder(ptr %rt, i32 %left, i32 0, i64 0, i32 0, i64 0, i64 %size)
; CHECK: icmp ne i32 %has_right, 0
; CHECK: br i1
; CHECK: call void @gicc.runtime.put_no_db.placeholder(ptr %rt, i32 %right, i32 0, i64 0, i32 0, i64 0, i64 %size)
