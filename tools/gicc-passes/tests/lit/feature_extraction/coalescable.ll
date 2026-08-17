; Can a loop's transfers be issued as one larger transfer?
;
; Only if consecutive iterations land exactly `size` apart at BOTH ends.
; Three fixtures, differing only in the offset expression:
;
;   contiguous   off = i * bytes,     size = bytes   -> mergeable
;   strided      off = i * stride,    size = bytes   -> NOT (gaps between)
;   halfsize     off = i * bytes,     size = 4096    -> NOT (stride unknown
;                                                      to equal the size)
;
; The two negatives matter more than the positive. Merging transfers that
; are not actually adjacent does not produce a slow program, it produces a
; wrong one: the merged transfer writes over whatever sits in the gap. So
; the analysis has to answer "no" whenever it cannot prove "yes", and
; comparison is syntactic on the expressions for exactly that reason.
;
; RUN: rm -rf %t.metadir && mkdir -p %t.metadir
; RUN: cp %S/../Inputs/k_coalesce_meta.json \
; RUN:    %t.metadir/_Z10k_coalescePN4gicc9DeviceCtxEiiill.json
; RUN: env GICC_MODE=feature-extract GICC_META_DIR=%t.metadir \
; RUN:                               GICC_FEATURES_OUT=%t.features.json \
; RUN:     %opt -load-pass-plugin=%gicc_passes_so \
; RUN:          -passes='gicc-feature-extraction' \
; RUN:          -disable-output %s 2>&1 | %FileCheck %s --check-prefix=STDERR
; RUN: cat %t.features.json | %FileCheck %s --check-prefix=JSON
; Exactly one of the three may claim it. Counted rather than pattern-matched
; because the JSON keys are emitted in alphabetical order, so a false claim
; would sit far from the site_id that produced it.
; RUN: grep -c '"coalescable": true'  %t.features.json | %FileCheck %s --check-prefix=ONETRUE
; RUN: grep -c '"coalescable": false' %t.features.json | %FileCheck %s --check-prefix=TWOFALSE

target triple = "x86_64-unknown-linux-gnu"

@.str.gicc = private unnamed_addr constant [17 x i8] c"gicc.launch_site\00", section "llvm.metadata"
@.str.f    = private unnamed_addr constant [12 x i8] c"launch.hpp\00\00", section "llvm.metadata"

@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }]
  [{ ptr, ptr, ptr, i32, ptr }
   { ptr @_ZN4gicc6launchITnDaXadL_Z10k_coalescePN4gicc9DeviceCtxEiiillEEEvRNS_7RuntimeE, ptr @.str.gicc, ptr @.str.f, i32 11, ptr null }],
   section "llvm.metadata"

define linkonce_odr void @_ZN4gicc6launchITnDaXadL_Z10k_coalescePN4gicc9DeviceCtxEiiillEEEvRNS_7RuntimeE(ptr %rt) {
  ret void
}
define void @main(ptr %rt) {
  call void @_ZN4gicc6launchITnDaXadL_Z10k_coalescePN4gicc9DeviceCtxEiiillEEEvRNS_7RuntimeE(ptr %rt)
  ret void
}

; STDERR: [feature-extract] wrote

; site 0: off = i*bytes, size = bytes -> the strides match the size
; JSON-DAG: "site_id": "t.cpp:5:k_coalesce::0"
; JSON-DAG: "coalescable": true

; sites 1 and 2 must both refuse. Exactly one site may claim it.
; JSON-DAG: "site_id": "t.cpp:6:k_coalesce::1"
; JSON-DAG: "site_id": "t.cpp:7:k_coalesce::2"

; ONETRUE: 1
; TWOFALSE: 2
