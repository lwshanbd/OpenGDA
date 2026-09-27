; A store that may reach heap memory is checked against [lo, hi) and, when
; inside, repeated at the address the mirror function returns. The range is
; read once, at entry.
;
; RUN: env GICC_MODE=chunk-lower %opt -load-pass-plugin=%gicc_passes_so \
; RUN:   -passes=gicc-nodb-mirror -S %s | %FileCheck %s

target triple = "nvptx64-nvidia-cuda"

; CHECK: @ompx__nodb = weak protected global [288 x i8] zeroinitializer

define void @k(ptr %v, i64 %i, float %x) {
; CHECK-LABEL: define void @k(
; CHECK:       %[[LO:.*]] = load ptr, ptr @ompx__nodb
; CHECK:       %[[HI:.*]] = load ptr, ptr getelementptr inbounds (i8, ptr @ompx__nodb, i64 8)
; CHECK:       br i1 %{{.*}}, label %[[SLOW:.*]], label %[[JOIN:.*]]
; CHECK:     [[SLOW]]:
; CHECK:       %[[Q:.*]] = call ptr @__gicc_nodb_mirror(ptr %{{.*}}, i64 4)
; CHECK:       icmp ne ptr %[[Q]], null
; CHECK:       store float %x, ptr %[[Q]], align 4
; CHECK:       store float %x, ptr %{{.*}}, align 4
; CHECK-NOT:   __gicc_nodb_mirror
; CHECK:       ret void
  %p = getelementptr float, ptr %v, i64 %i
  store float %x, ptr %p, align 4
  ret void
}

; The mirror function: finds the entry, poisons a store across its edge,
; marks whole words with the epoch, returns the peer address.
; CHECK-LABEL: define internal ptr @__gicc_nodb_mirror(ptr %0, i64 %1)
; CHECK-SAME:  #[[ATTR:[0-9]+]]
; CHECK:       load i32, ptr getelementptr inbounds (i8, ptr @ompx__nodb, i64 16)
; CHECK:       load i32, ptr getelementptr inbounds (i8, ptr @ompx__nodb, i64 20)
; CHECK:     poison:
; CHECK-NEXT:  store i32 1, ptr getelementptr inbounds (i8, ptr @ompx__nodb, i64 24)
; CHECK-NEXT:  ret ptr null
; CHECK: attributes #[[ATTR]] = { cold noinline nounwind "gicc-nodb" }
