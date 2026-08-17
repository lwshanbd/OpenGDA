#!/bin/bash
# run_transform_sites.sh - measure every legal transformation at every call
# site, so a plan can be scored against a real per-site oracle.
#
# THE TRAIN/TEST SPLIT IS DECLARED HERE, BEFORE ANY OF THIS IS MEASURED,
# and it is mechanical: every third site is held out. The first version of
# this experiment split by hand and the three sites with real headroom all
# landed in training, which left the held-out set too flat to separate any
# decider from any other. Re-splitting after seeing which sites are flat
# would have produced a better-looking number and no information, so the
# rule is fixed in the script instead of chosen later.
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

# Sites span message size, trip count, adjacency and how much independent
# work is available to hide the transfer behind. Roughly two thirds are
# provably adjacent, since a program where nothing can be merged has no
# transformation decision to make and one where everything can has no
# legality decision.
#
# name  bytes    ops  stride_mult  dist_us      (stride_mult 1 = adjacent)
SITES="
S01     256      64   1            0
S02     256      64   1            400
S03     1024     64   1            0
S04     4096     64   1            0
S05     4096     64   1            100
S06     16384    32   1            0
S07     16384    32   1            400
S08     65536    32   1            0
S09     65536    16   1            100
S10     262144   16   1            0
S11     256      64   2            0
S12     1024     64   2            0
S13     4096     32   2            0
S14     4096     32   2            400
S15     16384    32   2            0
S16     65536    16   2            0
"

# srun reads stdin, so inside a `while read` loop it swallows the rest of
# the site list and only the first one ever runs. Feed the loop on fd 3 and
# give srun /dev/null.
while read -r name bytes ops sm dist <&3; do
    [ -z "${name:-}" ] && continue
    echo "### site $name: bytes=$bytes ops=$ops stride_mult=$sm dist=${dist}us"
    GICC_SKIP_DWQ_INIT=1 GICC_PROXY_ENABLED=1 \
        timeout "$RUN_TIMEOUT" $SRUN "$BIN" \
            --site="$name" --bytes="$bytes" --ops="$ops" \
            --stride-mult="$sm" --dist="$dist" \
            --samples="$SAMPLES" --warmup="$WARMUP" < /dev/null 2>&1 \
        | tee /dev/stderr | grep '^CSV,' | grep -v '^CSV,site,' >> "$OUT"
    echo "###   -> $(grep -c '^CSV,' "$OUT") rows so far"
done 3<<< "$SITES"

echo
echo "wrote $(grep -c '^CSV,' "$OUT") rows to $OUT"
