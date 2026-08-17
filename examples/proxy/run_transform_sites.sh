#!/bin/bash
# run_transform_sites.sh - measure every legal transformation at every call
# site, so a plan can be scored against a real per-site oracle.
#
# The sites are chosen to disagree with each other. If they all preferred
# the same transformation there would be no decision to make and no reason
# for anything to choose per site.
#
#   A  tiny messages, contiguous      merging should pay the most
#   B  tiny messages, gaps            merging is ILLEGAL; only issue width
#   C  mid messages, contiguous       merging pays, less
#   D  large messages, contiguous     already near the bandwidth ceiling
#   E  tiny messages, contiguous, far  the wait is hidden anyway
#   F  mid messages, gaps             illegal again, different shape
#
#   ./examples/proxy/run_transform_sites.sh [outfile]
set -u

BIN=${BIN:-./build_ofi/examples/proxy/coalesce_bench}
OUT=${1:-docs/experiments/transform/sites.csv}
SAMPLES=${SAMPLES:-9}
WARMUP=${WARMUP:-3}
RUN_TIMEOUT=${RUN_TIMEOUT:-500}
SRUN="srun -p pci -N 2 -n 2 --ntasks-per-node=1 -c 8 --gpu-bind=none -t 8"

mkdir -p "$(dirname "$OUT")"
: > "$OUT"

# site  bytes    ops  stride_mult  dist_us
SITES="
A       256      64   1            0
B       256      64   2            0
C       4096     64   1            0
D       65536    32   1            0
E       256      64   1            400
F       4096     32   2            0
"

echo "$SITES" | while read -r name bytes ops sm dist; do
    [ -z "${name:-}" ] && continue
    echo "### site $name: bytes=$bytes ops=$ops stride_mult=$sm dist=${dist}us"
    GICC_SKIP_DWQ_INIT=1 GICC_PROXY_ENABLED=1 \
        timeout "$RUN_TIMEOUT" $SRUN "$BIN" \
            --site="$name" --bytes="$bytes" --ops="$ops" \
            --stride-mult="$sm" --dist="$dist" \
            --samples="$SAMPLES" --warmup="$WARMUP" 2>&1 \
        | tee /dev/stderr | grep '^CSV,' | grep -v '^CSV,site,' >> "$OUT"
done

echo
echo "wrote $(grep -c '^CSV,' "$OUT") rows to $OUT"
