; Phase B: the trace synthesizer turns a FieldLoad ArgRef into a
; gicc_runtime_host_mirror_of() call + byte-GEP + load at the host
; trace level. A FieldNotNull guard turns into a per-iteration
; icmp eq null inside the synthesized loop body that gates the
; placeholder call.
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/k_field_load_meta.json \
; RUN:    %t.metadir/_Z6k_haloPN4gicc9DeviceCtxEP4Desci.json
; RUN: env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-trace-synthesis' \
; RUN:          -S %s | %FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

@.str.gicc = private unnamed_addr constant [17 x i8]
    c"gicc.launch_site\00", section "llvm.metadata"
@.str.f    = private unnamed_addr constant [12 x i8]
    c"launch.hpp\00\00", section "llvm.metadata"

@llvm.global.annotations = appending global
  [1 x { ptr, ptr, ptr, i32, ptr }]
  [{ ptr, ptr, ptr, i32, ptr }
   { ptr @_ZN4gicc6launchITnDaXadL_Z6k_haloPN4gicc9DeviceCtxEP4DesciEEvRNS_7RuntimeEPS5_i,
     ptr @.str.gicc,
     ptr @.str.f,
     i32 47,
     ptr null }],
   section "llvm.metadata"

define linkonce_odr void @_ZN4gicc6launchITnDaXadL_Z6k_haloPN4gicc9DeviceCtxEP4DesciEEvRNS_7RuntimeEPS5_i(
    ptr %rt, ptr %transfers, i32 %num_transfers) {
  ret void
}

define void @main(ptr %rt, ptr %transfers, i32 %num_transfers) {
  call void @_ZN4gicc6launchITnDaXadL_Z6k_haloPN4gicc9DeviceCtxEP4DesciEEvRNS_7RuntimeEPS5_i(
      ptr %rt, ptr %transfers, i32 %num_transfers)
  ret void
}

; Launch site receives a trace call before the original launch wrapper.
; CHECK-LABEL: define void @main
; CHECK: call void @gicc_trace_k_halo(ptr %rt, ptr %transfers, i32 %num_transfers)

; The synthesized trace function: a host-side loop bounded by
; num_transfers; the loop body looks up the host mirror, computes the
; per-iter field address (iv * 24 + 16) for the FieldNotNull guard,
; loads the peer_recv_addr field as i8/ptr, compares to null, and only
; calls the placeholder for entries where the field is null.
; CHECK-LABEL: define internal void @gicc_trace_k_halo(ptr %rt, ptr %transfers, i32 %num_transfers)
; CHECK: phi i64
; CHECK: call ptr @gicc_runtime_host_mirror_of(ptr %rt, ptr %transfers)
; CHECK: %elem_off = mul i64 %iv, 24
; CHECK: %field_off = add i64 %elem_off, 16
; CHECK: getelementptr i8, ptr %host_mirror, i64 %field_off
; CHECK: load ptr, ptr %host_field_ptr
; CHECK: icmp eq ptr %host_field, null
; CHECK: br i1 %is_null, label %"guard.do.k.cpp:5:k_halo::0", label %"loop.latch.k.cpp:5:k_halo::0"
; CHECK: call void @gicc.runtime.put_no_db.placeholder
