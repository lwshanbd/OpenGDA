; The same model request must fail closed when the compiler metadata only has
; a dynamic bound/size.  A hint cannot assert its own legality proof.
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/k_loop_meta.json \
; RUN:    %t.metadir/_Z6k_loopPN4gicc9DeviceCtxEiiil.json
; RUN: not env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_coalesce_dynamic_loop.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-trace-synthesis' -disable-output %s 2>&1 | \
; RUN:     %FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

@.str.gicc = private unnamed_addr constant [17 x i8] c"gicc.launch_site\00", section "llvm.metadata"
@.str.f = private unnamed_addr constant [12 x i8] c"launch.hpp\00\00", section "llvm.metadata"

@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }]
  [{ ptr, ptr, ptr, i32, ptr }
   { ptr @_ZN4gicc6launchITnDaXadL_Z6k_loopPN4gicc9DeviceCtxEiiilEEvRNS_7RuntimeEiiiy,
     ptr @.str.gicc, ptr @.str.f, i32 47, ptr null }],
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

; CHECK: LLVM ERROR: gicc: COALESCE_LOOP rejected for site k.cpp:5:k_loop::0: requires a compiler-proven constant loop bound
