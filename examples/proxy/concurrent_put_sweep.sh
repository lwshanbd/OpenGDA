#!/bin/bash
# =============================================================================
# concurrent_put_sweep.sh
#
# In-kernel CONCURRENT-PUT scaling: a single rank-pair over a SINGLE CXI NIC,
# with K GPU blocks ("lanes") each issuing puts concurrently into the proxy
# ring(s). Tests whether GICC can keep multiple data transfers in flight at
# once (device-side concurrent submission), independent of multi-NIC scaling.
#
# bench_pingpong reports mean_us/msg AMORTIZED over all lanes
# (= wall / (batch_per_outer * lanes)). So:
#   - mean_us/msg DROPS as lanes rise  => concurrency scales (good)
#   - FLAT / rises                     => ring CAS contention or NIC-bound
#
# Two ring configurations:
#   GICC_NUM_PROXY_THREADS=1 : all lanes hash to ONE ring -> CAS contention on
#                              r->head + a single drain worker. Concurrency
#                              helps only to ~4 lanes, then regresses.
#   GICC_NUM_PROXY_THREADS=8 : lanes spread across 8 rings/workers -> contention
#                              free, scales monotonically to ~3.9x at 16 lanes.
#
# Result (2026-06-19, Tioga MI250X, cross-node, 256 B, mean_us/msg):
#   lanes:        1     2     4     8    16
#   NPT=1:      3.35  1.75  1.30  1.41  2.14   (single-ring ceiling at ~4 lanes)
#   NPT=8:      3.22  1.73  0.92  0.88  0.865  (multi-ring scales monotonically)
# Large messages (>=1 MB) are flat (~43 us/1MB, ~173 us/4MB) under every config:
# single-NIC bandwidth bound, concurrency and ring count do not matter there.
#
# Takeaway: GICC DOES transfer multiple data groups simultaneously and scales
# small-message rate ~3.9x with in-kernel concurrency -- but ONLY when
# GICC_NUM_PROXY_THREADS >= the lane count. The default single ring caps at
# ~4 concurrent producers due to the atomic_push CAS on the ring head.
#
# Usage: ./concurrent_put_sweep.sh
# Build first: GICC_ROOT=/p/lustre2/shan4/new-gicc bash build_bench_pingpong.sh
# =============================================================================
set -u
export LD_LIBRARY_PATH=/opt/rocm-6.4.0/lib:${LD_LIBRARY_PATH:-}
B="${GICC_ROOT:-/p/lustre2/shan4/new-gicc}/build_ofi/bench_pingpong"
LANES="${LANES:-1 2 4 8 16}"
NPTS="${NPTS:-1 8}"

for npt in ${NPTS}; do
  echo "######## GICC_NUM_PROXY_THREADS=${npt} (rings/workers) ########"
  for L in ${LANES}; do
    echo "---- lanes=${L} ----"
    GICC_NUM_PROXY_THREADS=${npt} \
      flux run -N2 -n2 -g1 -o mpibind=off "$B" --mode=proxy --lanes=${L} 2>&1 \
      | grep -E "^(256B|4KB|64KB|1MB|4MB)"
  done
done
echo "=== concurrent-put sweep complete ==="
