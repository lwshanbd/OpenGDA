#!/bin/bash
# tools/gicc-passes/tests/sweep/sweep_streams.sh
# Streams sweep on minimod, jacobi, mm_minimal.
# Sweep: streams in {1,2,4,8}, 2 grid sizes per workload, 3 reps each.
# W is fixed at 8 (the static default).
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
DATA_CSV="${GICC_ROOT}/tools/gicc-passes/tests/sweep/data.csv"
EXTRACT="${GICC_ROOT}/tools/gicc-passes/tests/sweep/extract_latency.py"

if [[ ! -f "${DATA_CSV}" ]]; then
    echo "workload,grid,ranks,W,streams,run_idx,latency_ms" > "${DATA_CSV}"
fi

# Build minimod via LTO once
META_DIR="${GICC_ROOT}/build_ofi/integration_scratch/sweep_meta"
rm -rf "${META_DIR}"; mkdir -p "${META_DIR}"
cd "${GICC_ROOT}/benchmarks/Minimod_MPI"
make TARGET=hip_gicc COMPILER=hipcc_gicc GICC_USE_LTO_PASS=1 clean >/dev/null
GICC_MODE=lower GICC_META_DIR="${META_DIR}" \
    make TARGET=hip_gicc COMPILER=hipcc_gicc GICC_USE_LTO_PASS=1 >/dev/null 2>&1

# Build jacobi & mm_minimal
cd "${GICC_ROOT}/build_ofi"
make -j jacobi mm_minimal >/dev/null 2>&1 || true

run_minimod () {
    local grid=$1 streams=$2 run=$3
    out=$(PMI_MAX_KVS_ENTRIES=512 FI_MR_CACHE_MAX_COUNT=0 GICC_STREAMS_MAX=${streams} \
          srun -p pci -N 4 -n 32 --ntasks-per-node=8 -t 2 \
          "${GICC_ROOT}/benchmarks/Minimod_MPI/run.sh" \
          "${GICC_ROOT}/benchmarks/Minimod_MPI/main_hip_gicc_hipcc_gicc" \
          --ngpus 32 --grid ${grid} --nsteps 100 2>&1) || true
    echo "${out}" | "${EXTRACT}" minimod ${grid} 32 8 ${streams} ${run}
}

# jacobi CLI: -nx <int> -ny <int> -niter <int>
# Output (rank 0, no -csv): "Done: N iters in Y s, final l2=Z"
run_jacobi () {
    local grid=$1 streams=$2 run=$3
    out=$(FI_MR_CACHE_MAX_COUNT=0 GICC_STREAMS_MAX=${streams} \
          srun -p pci -N 2 -n 16 --ntasks-per-node=8 -t 2 \
          "${GICC_ROOT}/build_ofi/examples/ofi/jacobi" \
          -nx ${grid} -ny ${grid} -niter 100 2>&1) || true
    echo "${out}" | "${EXTRACT}" jacobi ${grid} 16 8 ${streams} ${run}
}

# mm_minimal CLI: positional argv[1] = N (matrix dimension)
# Output (rank 0 stdout): "gicc::launch average (runs 2-9): Y us"
run_mm () {
    local grid=$1 streams=$2 run=$3
    out=$(FI_MR_CACHE_MAX_COUNT=0 GICC_STREAMS_MAX=${streams} \
          srun -p pci -N 2 -n 16 --ntasks-per-node=8 -t 2 \
          "${GICC_ROOT}/build_ofi/examples/ofi/mm_minimal" \
          ${grid} 2>&1) || true
    echo "${out}" | "${EXTRACT}" mm_minimal ${grid} 16 8 ${streams} ${run}
}

# minimod: grid 500 (medium), 1000 (large)
for grid in 500 1000; do
    for streams in 1 2 4 8; do
        for run in 1 2 3; do
            row=$(run_minimod ${grid} ${streams} ${run})
            echo "minimod ${grid} streams=${streams} run=${run}: ${row}"
            echo "${row}" >> "${DATA_CSV}"
        done
    done
done

# jacobi: grid 512 (medium), 1024 (large)
for grid in 512 1024; do
    for streams in 1 2 4 8; do
        for run in 1 2 3; do
            row=$(run_jacobi ${grid} ${streams} ${run})
            echo "jacobi ${grid} streams=${streams} run=${run}: ${row}"
            echo "${row}" >> "${DATA_CSV}"
        done
    done
done

# mm_minimal: N 2048, 4096
for grid in 2048 4096; do
    for streams in 1 2 4 8; do
        for run in 1 2 3; do
            row=$(run_mm ${grid} ${streams} ${run})
            echo "mm_minimal ${grid} streams=${streams} run=${run}: ${row}"
            echo "${row}" >> "${DATA_CSV}"
        done
    done
done

echo "=== sweep_streams done; CSV at ${DATA_CSV} ==="
