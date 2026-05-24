; Phase B: a kernel that loads from a host-mirrored device array should
; be marked HK by the HK Analysis pass (i.e. no "not host-knowable"
; warning on the put_no_db arguments). Same kernel as
; discovery/host_mirror_annotation.ll.
;
; RUN: env GICC_MODE=feature-extract \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-hk-analysis' \
; RUN:          -disable-output %s 2>&1 | %FileCheck %s --allow-empty
;
; CHECK-NOT: warning: gicc::put_no_db
; CHECK-NOT: not host-knowable

target triple = "amdgcn-amd-amdhsa"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

%struct.Desc = type { i32, i32, i64, ptr }

declare void @_ZN4gicc3putEPN4gicc9DeviceCtxEiimimmi(
    ptr, i32, i32, i64, i32, i64, i64, i32)

@.str.ann  = private constant [34 x i8]
    c"gicc_kernel_host_mirror=transfers\00", section "llvm.metadata"
@.str.file = private constant [9 x i8]
    c"halo.cpp\00", section "llvm.metadata"

@llvm.global.annotations =
  appending global [1 x { ptr, ptr, ptr, i32, ptr }]
  [{ ptr, ptr, ptr, i32, ptr }
   { ptr @halo_kernel,
     ptr @.str.ann,
     ptr @.str.file,
     i32 7,
     ptr null }], section "llvm.metadata"

define amdgpu_kernel void @halo_kernel(
    ptr %ctx, ptr %transfers, i32 %num_transfers) {
entry:
  br label %loop.head

loop.head:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.continue ]
  %iv32 = trunc i64 %iv to i32
  %cmp = icmp slt i32 %iv32, %num_transfers
  br i1 %cmp, label %loop.body, label %loop.exit

loop.body:
  %p_addr = getelementptr inbounds %struct.Desc, ptr %transfers,
                                       i64 %iv, i32 3
  %addr   = load ptr, ptr %p_addr, align 8
  %not_null = icmp ne ptr %addr, null
  br i1 %not_null, label %loop.continue, label %do_put

do_put:
  %p_peer = getelementptr inbounds %struct.Desc, ptr %transfers,
                                       i64 %iv, i32 0
  %peer   = load i32, ptr %p_peer, align 4
  %p_db   = getelementptr inbounds %struct.Desc, ptr %transfers,
                                       i64 %iv, i32 1
  %db     = load i32, ptr %p_db, align 4
  %p_dof  = getelementptr inbounds %struct.Desc, ptr %transfers,
                                       i64 %iv, i32 2
  %dof    = load i64, ptr %p_dof, align 8
  call void @_ZN4gicc3putEPN4gicc9DeviceCtxEiimimmi(
      ptr %ctx, i32 %peer, i32 %db, i64 %dof,
      i32 0, i64 0, i64 64, i32 0)
  br label %loop.continue

loop.continue:
  %iv.next = add i64 %iv, 1
  br label %loop.head

loop.exit:
  ret void
}
