; A compiler-proved loop-invariant descriptor can be evaluated once and
; passed with the runtime loop count. The transform preserves N queued PUTs;
; it removes only the six caller-side arrays and their staging loop.
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/k_reuse_loop_meta.json \
; RUN:    %t.metadir/_Z7k_reusePN4gicc9DeviceCtxEiimi.json
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_reuse_loop_descriptor.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-trace-synthesis' -S %s | \
; RUN:     %FileCheck %s --check-prefix=TRACE
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_reuse_loop_descriptor.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-trace-synthesis,gicc-dispatch-lowering' -S %s | \
; RUN:     %FileCheck %s --check-prefix=LOWER

target triple = "x86_64-unknown-linux-gnu"

@.str.gicc = private unnamed_addr constant [17 x i8] c"gicc.launch_site\00", section "llvm.metadata"
@.str.f = private unnamed_addr constant [12 x i8] c"launch.hpp\00\00", section "llvm.metadata"

@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }]
  [{ ptr, ptr, ptr, i32, ptr }
   { ptr @_ZN4gicc6launchITnDaXadL_Z7k_reusePN4gicc9DeviceCtxEiimiEEEvRNS_7RuntimeEiimi,
     ptr @.str.gicc, ptr @.str.f, i32 10, ptr null }],
   section "llvm.metadata"

define linkonce_odr void @_ZN4gicc6launchITnDaXadL_Z7k_reusePN4gicc9DeviceCtxEiimiEEEvRNS_7RuntimeEiimi(
    ptr %rt, i32 %peer, i32 %buf, i64 %bytes, i32 %n) {
  ret void
}

define void @main(ptr %rt, i32 %peer, i32 %buf, i64 %bytes, i32 %n) {
  call void @_ZN4gicc6launchITnDaXadL_Z7k_reusePN4gicc9DeviceCtxEiimiEEEvRNS_7RuntimeEiimi(
      ptr %rt, i32 %peer, i32 %buf, i64 %bytes, i32 %n)
  ret void
}

; TRACE-LABEL: define internal void @gicc_trace_k_reuse
; TRACE-NOT: alloca
; TRACE-NOT: loop.head
; TRACE: %repeat.bound.positive = icmp sgt i32 %n, 0
; TRACE: %repeat.count = select i1 %repeat.bound.positive, i32 %n, i32 0
; TRACE: call void @gicc.runtime.put_no_db.repeated.placeholder(ptr %rt, i32 %repeat.count, i32 %peer, i32 %buf, i64 0, i32 %buf, i64 0, i64 %bytes), !gicc.site_id !{{[0-9]+}}, !gicc.communication_transform !{{[0-9]+}}
; TRACE-NOT: gicc.runtime.put_no_db.batched.placeholder
; TRACE: !{{[0-9]+}} = !{!"REUSE_LOOP_DESCRIPTOR"}

; LOWER-LABEL: define internal void @gicc_trace_k_reuse
; LOWER-NOT: alloca
; LOWER: call void @gicc_runtime_dwq_enqueue_repeated(ptr %rt, i32 %repeat.count, i32 %peer, i32 %buf, i64 0, i32 %buf, i64 0, i64 %bytes)
; LOWER-NOT: gicc.runtime.put_no_db.repeated.placeholder
; LOWER-NOT: call void @gicc_runtime_dwq_enqueue_batched
