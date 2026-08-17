#!/bin/bash
# run_minimod_point.sh — re-measure the intra-node copy at Minimod's actual
# working point, one configuration per srun.
#
# Minimod's halo is `4 * (ny+2*ly) * (nz+2*lz) * sizeof(float)` bytes, laid
# out with x slowest, so at grid 1000 it is a single CONTIGUOUS ~15.5 MB
# transfer, two per timestep. 16 MB is the last size in ipc_copy_sweep's
# list, which makes that row the one that decides whether the eight-knob
# copy space is worth wiring into the pass for this application.
#
# The row that matters most is `memcpy`: GICC's IPC path is a host-side
# hipMemcpyAsync, so the tuned copy kernels only buy something if they beat
# it. Comparing the kernel configurations against each other and not against
# memcpy would answer a question nobody asked.
#
#   ./examples/proxy/run_minimod_point.sh [out.csv]
set -u
OUT="${1:-/tmp/minimod_point.csv}"
BIN=/p/lustre2/shan4/new-gicc/build_ofi/ipc_copy_sweep
SRUN="srun -p pci -N 1 -n 2 --ntasks-per-node=2 --gpu-bind=none -t 3"

# label:MECH:VEC:NT:UNROLL:BLOCK:NSTREAM:GRID:FENCE:PULL
CONFIGS="
memcpy-push:memcpy:16:1:1:256:1:0:3:0
memcpy-pull:memcpy:16:1:1:256:1:0:3:1
bestglobal:kernel:16:1:1:256:1:0:0:1
oracle16M:kernel:16:1:4:1024:4:64:2:1
oracle16M-push:kernel:16:1:4:1024:4:64:2:0
oracle16M-1stream:kernel:16:1:4:1024:1:64:2:1
gicc-today-kernel:kernel:16:1:1:256:1:0:3:0
"

echo "label,bytes,us,GBps" > "$OUT"
while IFS=: read -r label mech vec nt unroll block nstream grid fence pull; do
    [ -z "${label:-}" ] && continue
    echo "=== $label mech=$mech vec=$vec nt=$nt unroll=$unroll block=$block nstream=$nstream grid=$grid fence=$fence pull=$pull"
    timeout 240 env GICC_PROXY_ENABLED=1 GICC_SKIP_DWQ_INIT=1 \
        GICC_CP_MECH="$mech" GICC_CP_VEC="$vec" GICC_CP_NT="$nt" \
        GICC_CP_UNROLL="$unroll" GICC_CP_BLOCK="$block" \
        GICC_CP_NSTREAM="$nstream" GICC_CP_GRID="$grid" \
        GICC_CP_FENCE="$fence" GICC_CP_PULL="$pull" \
        $SRUN "$BIN" < /dev/null 2>&1 \
      | awk -v L="$label" -F, '/^[0-9]/ {print L","$1","$2","$3}' \
      | tee -a "$OUT"
done <<< "$CONFIGS"

echo "=== wrote $OUT"
