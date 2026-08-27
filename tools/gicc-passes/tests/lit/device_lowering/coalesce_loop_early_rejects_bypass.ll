; REQUIRES: gicc_lowering
; Dominance alone is insufficient: if a path from the communication-loop exit
; can bypass the original flush, moving the trigger to the exit would execute
; communication on a path where the program did not.  Reject unless the later
; flush also post-dominates the exit.
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: not env GICC_MODE=lower GICC_META_DIR=%t.metadir \
; RUN:     GICC_HINT_IN=%S/../Inputs/hint_device_coalesce_early_bypass.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-device-lowering' -disable-output %s 2>&1 | \
; RUN:     %FileCheck %s

target triple = "amdgcn-amd-amdhsa"

declare void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
    ptr, i32, i32, i64, i32, i64, i64)
declare void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr)

define amdgpu_kernel void @kernel_early_bypass(
    ptr %ctx, i32 %peer, i32 %buf, i1 %take_flush) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %next, %loop ]
  %off = mul i64 %iv, 4096
  call void @_ZN4gicc9put_no_dbEPN4gicc9DeviceCtxEiimimm(
      ptr %ctx, i32 %peer, i32 %buf, i64 %off,
      i32 %buf, i64 %off, i64 4096)
  %next = add nuw nsw i64 %iv, 1
  %more = icmp ult i64 %next, 16
  br i1 %more, label %loop, label %after

after:
  br i1 %take_flush, label %flush, label %done

flush:
  call void @_ZN4gicc5flushEPN4gicc9DeviceCtxE(ptr %ctx)
  br label %done

done:
  ret void
}

; CHECK: LLVM ERROR: gicc: COALESCE_LOOP_EARLY rejected for site ?:?:kernel_early_bypass::0: the later flush is not a mandatory post-dominated completion
