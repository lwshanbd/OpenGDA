; How wide a vector load may a copy of this transfer legally use?
;
; The bound is the largest power of two dividing the transfer size, the
; base offsets, and the loop stride -- a wide load has to stay inside one
; contiguous run and start aligned, and the stride is where the next gap
; begins.
;
; This is a legality bound, not a preference. Measured on a strided halo
; face, a vector width past the contiguous run does not run slowly: it is
; a GPU memory fault, in every orientation. So the interesting cases are
; the ones where the analysis must REFUSE, and four of the five below are
; exactly that.
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/k_vecmax_meta.json %t.metadir/_Z8k_vecmaxPN4gicc9DeviceCtxEiiill.json
; RUN: env GICC_MODE=feature-extract GICC_META_DIR=%t.metadir \
; RUN:                               GICC_FEATURES_OUT=%t.features.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-feature-extraction' \
; RUN:          -disable-output %s 2>&1 | %FileCheck %s --check-prefix=STDERR
; RUN: cat %t.features.json | %FileCheck %s --check-prefix=JSON

target triple = "x86_64-unknown-linux-gnu"

@.str.gicc = private unnamed_addr constant [17 x i8] c"gicc.launch_site\00", section "llvm.metadata"
@.str.f    = private unnamed_addr constant [12 x i8] c"launch.hpp\00\00", section "llvm.metadata"

@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }]
  [{ ptr, ptr, ptr, i32, ptr }
   { ptr @_ZN4gicc6launchITnDaXadL_Z8k_vecmaxPN4gicc9DeviceCtxEiiillEEEvRNS_7RuntimeE, ptr @.str.gicc, ptr @.str.f, i32 11, ptr null }],
   section "llvm.metadata"

define linkonce_odr void @_ZN4gicc6launchITnDaXadL_Z8k_vecmaxPN4gicc9DeviceCtxEiiillEEEvRNS_7RuntimeE(ptr %rt) {
  ret void
}
define void @main(ptr %rt) {
  call void @_ZN4gicc6launchITnDaXadL_Z8k_vecmaxPN4gicc9DeviceCtxEiiillEEEvRNS_7RuntimeE(ptr %rt)
  ret void
}

; STDERR: [feature-extract] wrote

; site 0: 4096B, 4096-aligned, contiguous -> the 16B cap applies
; JSON-DAG: "site_id": "t.cpp:5:k_vecmax::0"
; site 1: 12B transfers -- 16 does not divide 12
; JSON-DAG: "site_id": "t.cpp:6:k_vecmax::1"
; site 2: base offset by 2 bytes -- alignment bounds it, not size
; JSON-DAG: "site_id": "t.cpp:7:k_vecmax::2"
; site 3: size is a kernel formal -- nothing folds, so refuse
; JSON-DAG: "site_id": "t.cpp:8:k_vecmax::3"
; site 4: 64B payload every 4096B -- the run is 64B, which still
; admits a 16B load, so the cap applies rather than the gap
; JSON-DAG: "site_id": "t.cpp:9:k_vecmax::4"
;
; Two 16, one 4, one 2, one 1. Counted rather than matched in place
; because the JSON keys are alphabetical, so the width sits far from
; the site_id that produced it -- and counted WITH the trailing comma,
; because otherwise a search for 1 also matches inside 16.
; because the JSON keys are alphabetical so the width sits far from the
; site_id that produced it.
; RUN: grep -c '"max_vector_bytes": 16,' %t.features.json | %FileCheck %s --check-prefix=N16
; RUN: grep -c '"max_vector_bytes": 4,' %t.features.json | %FileCheck %s --check-prefix=N4
; RUN: grep -c '"max_vector_bytes": 2,' %t.features.json | %FileCheck %s --check-prefix=N2
; RUN: grep -c '"max_vector_bytes": 1,' %t.features.json | %FileCheck %s --check-prefix=N1
; N16: 2
; N4: 1
; N2: 1
; N1: 1
