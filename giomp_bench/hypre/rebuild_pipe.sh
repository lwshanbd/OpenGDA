#!/bin/bash
# rebuild_pipe.sh [SHIFT]: the GiOMP tree's struct driver with the pipelined
# pack (HYPRE_GIOMP_PIPELINE): struct_communication.c compiled through the
# gicc-passes plugin, pieces of at least 2^SHIFT bytes (default 22).
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
H=/shared/data1/Users/l1065028/giomp_bench/hypre; SHIFT=${1:-22}
PASSES=/shared/data1/Users/l1065028/giomp_ib_test/passes-build/libgicc-passes.so
S=$H/src-giomp/src; OUT=$H/pipe$SHIFT; mkdir -p $OUT; rm -rf $OUT/meta; mkdir -p $OUT/meta
export GICC_MODE=chunk-lower GICC_META_DIR=$OUT/meta
LOG=$OUT/build.log
CC=$(grep -m1 -- "-c struct_communication.c" $H/rebuild-giomp.log | sed 's/ -c struct_communication.c -o struct_communication.obj//')
[ -n "$CC" ] || { echo "no compile line"; exit 1; }
cd $S/struct_mv && $CC -DHYPRE_GIOMP_PIPELINE -DOMPX_PIPE_PIECE_SHIFT=$SHIFT -foffload-lto -fpass-plugin=$PASSES \
   -c struct_communication.c -o $OUT/struct_communication.obj > $LOG 2>&1 || { grep -m10 -E "error|gicc" $LOG; exit 1; }
cp $S/hypre/lib/libHYPRE.a $OUT/libHYPRE.a && ar r $OUT/libHYPRE.a $OUT/struct_communication.obj
# ar matches by member name: make sure the pipelined object replaced the old one
ar t $OUT/libHYPRE.a | grep -c "^struct_communication.obj$"
LINK=$(grep -m1 "clang++  -o struct struct.obj" $H/rebuild-giomp.log | sed "s#-L$S/hypre/lib -lHYPRE#$OUT/libHYPRE.a#")
cd $S/test && $LINK -foffload-lto -fpass-plugin=$PASSES -o $OUT/struct >> $LOG 2>&1 || { grep -m10 -E "error|gicc" $LOG; exit 1; }
grep -E "gicc-" $LOG | sort | uniq -c | head -20
echo BUILT $OUT/struct
