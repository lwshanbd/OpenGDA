#!/bin/bash
# run_ipc_joint.sh - joint sweep of the intra-node copy knobs.
#
# The question this exists to answer is NOT "which configuration is
# fastest". It is whether the axes INTERACT: whether the best value of one
# knob changes when another is set differently. If they do not, the space
# decomposes into independent per-axis rules and a hand-written rule is
# sufficient, which is what four earlier decision spaces on the cross-node
# path turned out to be. If they do, no set of independent rules can
# express the optimum and a model has something to do.
#
# Each binary run sweeps all nine transfer sizes internally, so one run
# yields a whole size row and the size axis comes for free.
#
#   ./examples/proxy/run_ipc_joint.sh [outfile]
set -u

BIN=${BIN:-./build_ofi/ipc_copy_sweep}
OUT=${1:-docs/experiments/ipc/joint.csv}
ITERS=${ITERS:-100}
RUN_TIMEOUT=${RUN_TIMEOUT:-150}
SRUN="srun -p pci -t 3 -N 1 -n 2 --gpu-bind=none"

VECS=${VECS:-"1 4 8 16"}
NTS=${NTS:-"0 1"}
UNROLLS=${UNROLLS:-"1 2 4"}
BLOCK=${BLOCK:-256}
NSTREAM=${NSTREAM:-1}

mkdir -p "$(dirname "$OUT")"
echo "vec,nt,unroll,block,nstream,bytes,us,GBps" > "$OUT"

for V in $VECS; do
  for N in $NTS; do
    for U in $UNROLLS; do
      printf '### vec=%s nt=%s unroll=%s ' "$V" "$N" "$U"
      out=$(GICC_PROXY_ENABLED=1 GICC_SKIP_DWQ_INIT=1 \
            GICC_CP_VEC=$V GICC_CP_NT=$N GICC_CP_UNROLL=$U \
            GICC_CP_BLOCK=$BLOCK GICC_CP_NSTREAM=$NSTREAM \
            GICC_CP_ITERS=$ITERS \
            timeout "$RUN_TIMEOUT" $SRUN "$BIN" < /dev/null 2>&1)
      if echo "$out" | grep -q "Memory access fault"; then
          echo "ILLEGAL (gpu fault)"
          continue
      fi
      n=0
      while IFS=, read -r bytes us gbps; do
          case "$bytes" in ''|\#*) continue;; esac
          echo "$V,$N,$U,$BLOCK,$NSTREAM,$bytes,$us,$gbps" >> "$OUT"
          n=$((n+1))
      done <<< "$(echo "$out" | grep -E '^[0-9]+,')"
      echo "($n sizes)"
    done
  done
done

echo
echo "wrote $(($(wc -l < "$OUT") - 1)) rows to $OUT"
