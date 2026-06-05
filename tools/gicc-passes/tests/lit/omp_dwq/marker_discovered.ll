; Verify GICCOmpDeviceDiscovery finds omp_dwq marker calls inside an
; amdgpu_kernel that uses the __omp_offloading_* naming convention and
; writes a KernelTemplate JSON with hk_capable=true for the put site.
;
; The fixture IR is a trimmed version of the amdgcn device module produced
; by compiling examples/omp/gicc_omp_dwq.hpp with:
;   clang++ -fopenmp --offload-arch=gfx90a -O2 -save-temps
; The key feature: after O2 inlining the marker bodies become
; fastcc tail calls with the .internalized suffix; our pass must
; recognise them by substring ("7omp_dwq3put" / "7omp_dwq5flush").
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: env GICC_MODE=omp-dwq GICC_META_DIR=%t.metadir \
; RUN:   %opt -load-pass-plugin=%gicc_passes_so -passes='gicc-omp-device-discovery' \
; RUN:   -disable-output %s 2>&1 | %FileCheck %s --check-prefix=STDERR
; RUN: cat %t.metadir/*.json | %FileCheck %s --check-prefix=JSON

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

; The .internalized fastcc clones that survive -O2 inlining.
declare fastcc void @_ZN4gicc7omp_dwq3putEPNS_9DeviceCtxEiimimm.internalized(ptr, i32, i32, i64, i32, i64, i64)
declare fastcc void @_ZN4gicc7omp_dwq5flushEPNS_9DeviceCtxE.internalized(ptr)
declare i32 @__kmpc_target_init(ptr, ptr)
declare void @__kmpc_target_deinit()

; Kernel formals: %0=dyn_env_ptr, %1=ctx_ptr, %2..%7=i64 captures
define amdgpu_kernel void @__omp_offloading_14_7363c44__Z3runPN4gicc9DeviceCtxEiimimm_l3(
    ptr noalias noundef %0, ptr noundef %1,
    i64 noundef %2, i64 noundef %3, i64 noundef %4,
    i64 noundef %5, i64 noundef %6, i64 noundef %7) {
entry:
  %init = call i32 @__kmpc_target_init(ptr null, ptr %0)
  %is_main = icmp eq i32 %init, -1
  br i1 %is_main, label %body, label %exit

body:
  %peer    = trunc i64 %2 to i32
  %dst_buf = trunc i64 %3 to i32
  %src_buf = trunc i64 %5 to i32
  tail call fastcc void @_ZN4gicc7omp_dwq3putEPNS_9DeviceCtxEiimimm.internalized(
      ptr noundef %1, i32 noundef %peer, i32 noundef %dst_buf, i64 noundef %4,
      i32 noundef %src_buf, i64 noundef %6, i64 noundef %7)
  tail call fastcc void @_ZN4gicc7omp_dwq5flushEPNS_9DeviceCtxE.internalized(
      ptr noundef %1)
  call void @__kmpc_target_deinit()
  br label %exit

exit:
  ret void
}

; STDERR: [omp-dwq-discovery] kernel __omp_offloading_14_7363c44__Z3runPN4gicc9DeviceCtxEiimimm_l3
; STDERR-DAG: kind=put_no_db
; STDERR-DAG: kind=flush
; STDERR: wrote

; JSON: "kernel_mangled": "_Z3runPN4gicc9DeviceCtxEiimimm"
; JSON: "kind": "put_no_db"
; JSON: "hk_capable": true
; JSON: "kind": "flush"
