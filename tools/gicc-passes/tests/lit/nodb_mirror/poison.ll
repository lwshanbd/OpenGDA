; Writes the pass cannot repeat poison the epoch instead: an atomic or a
; memset that meets [lo, hi), and a call it cannot see into whenever any
; source is posted -- GICC's own device get among them, which writes through
; the proxy. An intrinsic that writes through a pointer counts as such a
; call. Trusted runtime calls are left alone.
;
; RUN: env GICC_MODE=chunk-lower %opt -load-pass-plugin=%gicc_passes_so \
; RUN:   -passes=gicc-nodb-mirror -S %s | %FileCheck %s

target triple = "nvptx64-nvidia-cuda"

declare void @opaque(ptr)
declare void @__kmpc_barrier(ptr, i32)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
declare void @llvm.masked.store.v4f32.p0(<4 x float>, ptr, i32, <4 x i1>)
declare void @ompx_get_dev(i32, ptr, ptr, i64)

define void @atomic(ptr %v) {
; CHECK-LABEL: define void @atomic(
; CHECK:       icmp ult i64 %{{.*}}, %nodb.hi
; CHECK:       icmp ugt i64 %{{.*}}, %nodb.lo
; CHECK:       store i32 1, ptr getelementptr inbounds (i8, ptr @ompx__nodb, i64 24)
; CHECK:       atomicrmw add ptr %v, i32 1 monotonic
  %old = atomicrmw add ptr %v, i32 1 monotonic
  ret void
}

define void @memset(ptr %v, i64 %n) {
; CHECK-LABEL: define void @memset(
; CHECK:       add i64 %{{.*}}, %n
; CHECK:       store i32 1, ptr getelementptr inbounds (i8, ptr @ompx__nodb, i64 24)
; CHECK:       call void @llvm.memset.p0.i64(ptr %v, i8 0, i64 %n, i1 false)
  call void @llvm.memset.p0.i64(ptr %v, i8 0, i64 %n, i1 false)
  ret void
}

define void @calls(ptr %v) {
; CHECK-LABEL: define void @calls(
; CHECK:       icmp ne i64 %nodb.lo, %nodb.hi
; CHECK:       store i32 1, ptr getelementptr inbounds (i8, ptr @ompx__nodb, i64 24)
; CHECK:       call void @opaque(ptr %v)
; CHECK-NOT:   store i32 1
; CHECK:       call void @__kmpc_barrier(ptr null, i32 0)
  call void @opaque(ptr %v)
  call void @__kmpc_barrier(ptr null, i32 0)
  ret void
}

define void @intrinsic(ptr %v, <4 x float> %x, <4 x i1> %m) {
; CHECK-LABEL: define void @intrinsic(
; CHECK:       icmp ne i64 %nodb.lo, %nodb.hi
; CHECK:       store i32 1, ptr getelementptr inbounds (i8, ptr @ompx__nodb, i64 24)
; CHECK:       call void @llvm.masked.store
  call void @llvm.masked.store.v4f32.p0(<4 x float> %x, ptr %v, i32 4, <4 x i1> %m)
  ret void
}

define void @get(ptr %d, ptr %s) {
; CHECK-LABEL: define void @get(
; CHECK:       store i32 1, ptr getelementptr inbounds (i8, ptr @ompx__nodb, i64 24)
; CHECK:       call void @ompx_get_dev
  call void @ompx_get_dev(i32 1, ptr %d, ptr %s, i64 64)
  ret void
}
