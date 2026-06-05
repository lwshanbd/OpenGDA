; Verify GICCOmpHostDiscovery recognizes a __tgt_target_kernel launch,
; resolves the device source-mangled name via the offloading entry, recovers
; the captured host scalars from the .offload_baseptrs array, and synthesizes
; a host trace call (reusing the trace-synthesis machinery) placed BEFORE the
; launch.
;
; The fixture IR is a trimmed version of the x86_64 host module produced by
; compiling examples/omp/gicc_omp_dwq.hpp with:
;   clang++ -fopenmp --offload-arch=gfx90a -O2
; for the probe TU:
;   void run(gicc::DeviceCtx* ctx,int peer,int dbuf,size_t doff,
;            int sbuf,size_t soff,size_t n) {
;     #pragma omp target is_device_ptr(ctx)
;     { gicc::omp_dwq::put(ctx,peer,dbuf,doff,sbuf,soff,n);
;       gicc::omp_dwq::flush(ctx); }
;   }
;
; The device discovery pass emits omp_run_meta.json (kernel key
; _Z3runPN4gicc9DeviceCtxEiimimm); we pre-seed the metadir with it.
;
; Capture -> device-param mapping (slot K of .offload_baseptrs -> device
; formal K+1):
;   slot 0 = ctx ptr            -> device param 1
;   slot 1 = zext(peer i32)     -> device param 2
;   slot 2 = zext(dbuf i32)     -> device param 3
;   slot 3 = doff i64           -> device param 4
;   slot 4 = zext(sbuf i32)     -> device param 5
;   slot 5 = soff i64           -> device param 6
;   slot 6 = n    i64           -> device param 7
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/omp_run_meta.json %t.metadir/_Z3runPN4gicc9DeviceCtxEiimimm.json
; RUN: env GICC_MODE=omp-dwq GICC_META_DIR=%t.metadir \
; RUN:   %opt -load-pass-plugin=%gicc_passes_so \
; RUN:     -passes='gicc-omp-host-discovery,gicc-trace-synthesis' \
; RUN:   -S %s | %FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ident_t = type { i32, i32, i32, i32, ptr }
%struct.__tgt_offload_entry = type { ptr, ptr, i64, i32, i32 }
%struct.__tgt_kernel_arguments = type { i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, i64, i64, [3 x i32], [3 x i32], i32 }

@.__omp_offloading_14_7363c44__Z3runPN4gicc9DeviceCtxEiimimm_l3.region_id = weak constant i8 0
@.offload_sizes = private unnamed_addr constant [7 x i64] [i64 8, i64 4, i64 4, i64 8, i64 4, i64 8, i64 8]
@.offload_maptypes = private unnamed_addr constant [7 x i64] [i64 288, i64 800, i64 800, i64 800, i64 800, i64 800, i64 800]
@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8
@.offloading.entry_name = internal unnamed_addr constant [62 x i8] c"__omp_offloading_14_7363c44__Z3runPN4gicc9DeviceCtxEiimimm_l3\00"
@.offloading.entry.__omp_offloading_14_7363c44__Z3runPN4gicc9DeviceCtxEiimimm_l3 = weak local_unnamed_addr constant %struct.__tgt_offload_entry { ptr @.__omp_offloading_14_7363c44__Z3runPN4gicc9DeviceCtxEiimimm_l3.region_id, ptr @.offloading.entry_name, i64 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1

; The trace call must be inserted BEFORE the launch and carry, in order:
;   rt(=live Runtime from gicc_runtime_current()), ctx ptr, peer, dbuf, doff,
;   sbuf, soff, n. arg0 is sourced from a call to the bridge C ABI
;   gicc_runtime_current(), NOT the ctx pointer stand-in.
; CHECK-LABEL: define dso_local void @_Z3runPN4gicc9DeviceCtxEiimimm(
; CHECK: %[[RT:.*]] = call ptr @gicc_runtime_current()
; CHECK: call void @gicc_trace__Z3runPN4gicc9DeviceCtxEiimimm(ptr %[[RT]],
; CHECK: call i32 @__tgt_target_kernel(

define dso_local void @_Z3runPN4gicc9DeviceCtxEiimimm(ptr noundef %0, i32 noundef %1, i32 noundef %2, i64 noundef %3, i32 noundef %4, i64 noundef %5, i64 noundef %6) {
  %8 = alloca [7 x ptr], align 8
  %9 = alloca [7 x ptr], align 8
  %10 = alloca %struct.__tgt_kernel_arguments, align 8
  %peer.ext = zext i32 %1 to i64
  %dbuf.ext = zext i32 %2 to i64
  %sbuf.ext = zext i32 %4 to i64
  store ptr %0, ptr %8, align 8
  store ptr %0, ptr %9, align 8
  %11 = getelementptr inbounds i8, ptr %8, i64 8
  store i64 %peer.ext, ptr %11, align 8
  %12 = getelementptr inbounds i8, ptr %9, i64 8
  store i64 %peer.ext, ptr %12, align 8
  %13 = getelementptr inbounds i8, ptr %8, i64 16
  store i64 %dbuf.ext, ptr %13, align 8
  %14 = getelementptr inbounds i8, ptr %9, i64 16
  store i64 %dbuf.ext, ptr %14, align 8
  %15 = getelementptr inbounds i8, ptr %8, i64 24
  store i64 %3, ptr %15, align 8
  %16 = getelementptr inbounds i8, ptr %9, i64 24
  store i64 %3, ptr %16, align 8
  %17 = getelementptr inbounds i8, ptr %8, i64 32
  store i64 %sbuf.ext, ptr %17, align 8
  %18 = getelementptr inbounds i8, ptr %9, i64 32
  store i64 %sbuf.ext, ptr %18, align 8
  %19 = getelementptr inbounds i8, ptr %8, i64 40
  store i64 %5, ptr %19, align 8
  %20 = getelementptr inbounds i8, ptr %9, i64 40
  store i64 %5, ptr %20, align 8
  %21 = getelementptr inbounds i8, ptr %8, i64 48
  store i64 %6, ptr %21, align 8
  %22 = getelementptr inbounds i8, ptr %9, i64 48
  store i64 %6, ptr %22, align 8
  store i32 3, ptr %10, align 8
  %23 = getelementptr inbounds i8, ptr %10, i64 4
  store i32 7, ptr %23, align 4
  %24 = getelementptr inbounds i8, ptr %10, i64 8
  store ptr %8, ptr %24, align 8
  %25 = getelementptr inbounds i8, ptr %10, i64 16
  store ptr %9, ptr %25, align 8
  %26 = getelementptr inbounds i8, ptr %10, i64 24
  store ptr @.offload_sizes, ptr %26, align 8
  %27 = getelementptr inbounds i8, ptr %10, i64 32
  store ptr @.offload_maptypes, ptr %27, align 8
  %30 = call i32 @__tgt_target_kernel(ptr nonnull @1, i64 -1, i32 -1, i32 0, ptr nonnull @.__omp_offloading_14_7363c44__Z3runPN4gicc9DeviceCtxEiimimm_l3.region_id, ptr nonnull %10)
  ret void
}

declare i32 @__tgt_target_kernel(ptr, i64, i32, i32, ptr, ptr) local_unnamed_addr
