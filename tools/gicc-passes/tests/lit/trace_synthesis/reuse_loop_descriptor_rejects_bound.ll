; The scalar repeated-descriptor ABI intentionally accepts only a signed-i32
; dynamic loop bound. A wider bound cannot be truncated by a model request.
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/k_reuse_loop_i64_meta.json \
; RUN:    %t.metadir/_Z7k_reusePN4gicc9DeviceCtxEiimi.json
; RUN: not env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_reuse_loop_descriptor.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-trace-synthesis' -disable-output %s 2>&1 | \
; RUN:     %FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

@.str.gicc = private unnamed_addr constant [17 x i8] c"gicc.launch_site\00", section "llvm.metadata"
@.str.f = private unnamed_addr constant [12 x i8] c"launch.hpp\00\00", section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }]
  [{ ptr, ptr, ptr, i32, ptr }
   { ptr @_ZN4gicc6launchITnDaXadL_Z7k_reusePN4gicc9DeviceCtxEiimiEEEvRNS_7RuntimeEiimm,
     ptr @.str.gicc, ptr @.str.f, i32 10, ptr null }],
   section "llvm.metadata"

define linkonce_odr void @_ZN4gicc6launchITnDaXadL_Z7k_reusePN4gicc9DeviceCtxEiimiEEEvRNS_7RuntimeEiimm(
    ptr %rt, i32 %peer, i32 %buf, i64 %bytes, i64 %n) {
  ret void
}

define void @main(ptr %rt, i32 %peer, i32 %buf, i64 %bytes, i64 %n) {
  call void @_ZN4gicc6launchITnDaXadL_Z7k_reusePN4gicc9DeviceCtxEiimiEEEvRNS_7RuntimeEiimm(
      ptr %rt, i32 %peer, i32 %buf, i64 %bytes, i64 %n)
  ret void
}

; CHECK: LLVM ERROR: gicc: REUSE_LOOP_DESCRIPTOR rejected for site reuse.cpp:5:k_reuse::0: dynamic loop bound must be an i32 kernel formal
