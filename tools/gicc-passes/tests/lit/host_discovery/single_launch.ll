; GICCHostDiscovery walks @llvm.global.annotations, finds the
; gicc.launch_site-annotated wrapper, decodes the kernel mangled name
; from the wrapper's NTTP (XadL_Z<kernel>E), and loads the matching
; per-kernel meta JSON from GICC_META_DIR.
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/halo_kernel_meta.json \
; RUN:    %t.metadir/_Z11halo_kernelPN4gicc9DeviceCtxEiiimmmmmii.json
; RUN: env GICC_MODE=feature-extract GICC_META_DIR=%t.metadir \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-host-discovery' \
; RUN:          -disable-output %s 2>&1 | %FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

%struct.dim3 = type { i32, i32, i32 }

@.str.gicc = private unnamed_addr constant [17 x i8] c"gicc.launch_site\00", section "llvm.metadata"
@.str.f    = private unnamed_addr constant [12 x i8] c"launch.hpp\00\00", section "llvm.metadata"

@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }]
  [{ ptr, ptr, ptr, i32, ptr }
   { ptr @_ZN4gicc6launchITnDaXadL_Z11halo_kernelPN4gicc9DeviceCtxEiiimmmmmiiEEvRNS_7RuntimeE,
     ptr @.str.gicc,
     ptr @.str.f,
     i32 47,
     ptr null }],
   section "llvm.metadata"

; A stand-in for the gicc::launch<&halo_kernel> instantiation. We don't
; care about the body for discovery — only that the symbol exists, has
; the kernel-pointer NTTP encoded in its name, and is callable.
define linkonce_odr void @_ZN4gicc6launchITnDaXadL_Z11halo_kernelPN4gicc9DeviceCtxEiiimmmmmiiEEvRNS_7RuntimeE(ptr %rt) {
  ret void
}

define void @main(ptr %rt) {
entry:
  call void @_ZN4gicc6launchITnDaXadL_Z11halo_kernelPN4gicc9DeviceCtxEiiimmmmmiiEEvRNS_7RuntimeE(ptr %rt)
  ret void
}

; CHECK: [host-discovery] found 1 launch site(s) for kernel halo_kernel
