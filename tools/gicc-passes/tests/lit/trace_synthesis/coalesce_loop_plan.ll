; A model-selected structural plan may only name a compiler-generated
; candidate.  Here the metadata proves a four-trip PUT loop with 1024-byte
; source and destination strides, so TraceSynthesis may replace its four
; descriptors by one 4096-byte placeholder.  No source or model-authored IR
; participates in the transform.
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/k_coalesce_constant_meta.json \
; RUN:    %t.metadir/_Z6k_planPN4gicc9DeviceCtxEii.json
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_coalesce_loop.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-trace-synthesis' -S %s | \
; RUN:     %FileCheck %s --check-prefix=TRACE
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_coalesce_loop.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-trace-synthesis,gicc-dispatch-lowering' -S %s | \
; RUN:     %FileCheck %s --check-prefix=LOWER

target triple = "x86_64-unknown-linux-gnu"

@.str.gicc = private unnamed_addr constant [17 x i8] c"gicc.launch_site\00", section "llvm.metadata"
@.str.f = private unnamed_addr constant [12 x i8] c"launch.hpp\00\00", section "llvm.metadata"

@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }]
  [{ ptr, ptr, ptr, i32, ptr }
   { ptr @_ZN4gicc6launchITnDaXadL_Z6k_planPN4gicc9DeviceCtxEiiEEEvRNS_7RuntimeEii,
     ptr @.str.gicc, ptr @.str.f, i32 10, ptr null }],
   section "llvm.metadata"

define linkonce_odr void @_ZN4gicc6launchITnDaXadL_Z6k_planPN4gicc9DeviceCtxEiiEEEvRNS_7RuntimeEii(
    ptr %rt, i32 %peer, i32 %buf) {
  ret void
}

define void @main(ptr %rt, i32 %peer, i32 %buf) {
  call void @_ZN4gicc6launchITnDaXadL_Z6k_planPN4gicc9DeviceCtxEiiEEEvRNS_7RuntimeEii(
      ptr %rt, i32 %peer, i32 %buf)
  ret void
}

; TRACE-LABEL: define internal void @gicc_trace_k_plan
; TRACE-NOT: alloca
; TRACE-NOT: loop.head
; TRACE: call void @gicc.runtime.put_no_db.placeholder(ptr %rt, i32 %peer, i32 %buf, i64 0, i32 %buf, i64 0, i64 4096), !gicc.site_id !{{[0-9]+}}, !gicc.communication_transform !{{[0-9]+}}
; TRACE-NOT: gicc.runtime.put_no_db.batched.placeholder
; TRACE: ret void
; TRACE: !{{[0-9]+}} = !{!"COALESCE_LOOP"}

; LOWER-LABEL: define internal void @gicc_trace_k_plan
; LOWER-NOT: alloca
; LOWER: call void @gicc_runtime_dwq_enqueue(ptr %rt, i32 %peer, i32 %buf, i64 0, i32 %buf, i64 0, i64 4096)
; LOWER-NOT: gicc_runtime_dwq_enqueue_batched
; LOWER: ret void
