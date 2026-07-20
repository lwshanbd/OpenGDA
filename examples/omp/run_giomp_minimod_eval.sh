#!/bin/bash
# Strong-scale GiOMP, DiOMP, and GPU-aware MPI Minimod on Tioga.
# Each node contributes eight ranks and eight GPUs.  Every configuration is
# launched independently at top level because nested Flux allocations hide the
# complete device set required by GiOMP's same-node IPC path.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
MM_DIOMP="${GIOMP_MINIMOD_DIOMP_ROOT:-/p/lustre2/shan4/DiOMP/benchmarks/benchmarks/Minimod_DiOMP}"
MM_MPI="${GIOMP_MINIMOD_MPI_ROOT:-/p/lustre2/shan4/DiOMP/benchmarks/benchmarks/Minimod_MPI}"
GICC_BIN="${MM_DIOMP}/main_omp_gicc_clang21"
DIOMP_BIN="${MM_DIOMP}/main_omp_offload_opt-gpu_multi_nowait_clang"
MPI_BIN="${MM_MPI}/main_omp_offload_opt-gpu_multi_nowait_clang"
MPI_WRAP="${MM_MPI}/run_gpu_wrap.sh"

OUT="${1:-${GICC_ROOT}/build_ofi/minimod_paper}"
QUEUE="${GIOMP_QUEUE:-pdebug}"
GRID="${GIOMP_MINIMOD_GRID:-1200}"
RUNS="${GIOMP_MINIMOD_RUNS:-3}"
MAX_TRIES="${GIOMP_MINIMOD_MAX_TRIES:-4}"
NODES_SPEC="${GIOMP_MINIMOD_NODES:-1 2 3 4 5 6 7 8}"
BACKENDS_SPEC="${GIOMP_MINIMOD_BACKENDS:-giomp-no-overlap giomp-overlap diomp mpi}"
read -r -a NODES <<<"${NODES_SPEC}"
read -r -a BACKENDS <<<"${BACKENDS_SPEC}"

for executable in "${GICC_BIN}" "${DIOMP_BIN}" "${MPI_BIN}" "${MPI_WRAP}"; do
    if [[ ! -x "${executable}" ]]; then
        echo "missing executable: ${executable}" >&2
        exit 2
    fi
done
if [[ ! "${RUNS}" =~ ^[1-9][0-9]*$ || ! "${MAX_TRIES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GIOMP_MINIMOD_RUNS and GIOMP_MINIMOD_MAX_TRIES must be positive integers" >&2
    exit 2
fi
for nodes in "${NODES[@]}"; do
    if [[ ! "${nodes}" =~ ^[1-8]$ ]]; then
        echo "node counts must be integers from 1 through 8: ${nodes}" >&2
        exit 2
    fi
done

source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
module use /g/g91/shan4/modulefiles
module load diomp/1.0 >/dev/null 2>&1

export TMPDIR=/tmp
export FI_MR_CACHE_MAX_COUNT=0
export PMI_MAX_KVS_ENTRIES=512
export LD_LIBRARY_PATH="/p/lustre2/shan4/softwares/diomp/lib:/p/lustre2/shan4/softwares/diomp/lib/x86_64-unknown-linux-gnu:/opt/rocm-6.4.3/lib:/opt/cray/pe/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p "${OUT}"

run_case() {
    local backend="$1" nodes="$2" gpus="$3" trial="$4" log="$5"
    local common=(flux run -q "${QUEUE}" -t 10m -N "${nodes}" -n "${gpus}"
                  -g1 -o mpibind=off -l)
    case "${backend}" in
        giomp-no-overlap)
            HSA_XNACK=1 GICC_PROXY_ENABLED=1 GICC_SKIP_DWQ_INIT=1 \
            GICC_HALO_IPC=1 GICC_HALO_OVERLAP=0 \
            ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
                "${common[@]}" "${GICC_BIN}" --ngpus "${gpus}" --grid "${GRID}" \
                >"${log}" 2>&1
            ;;
        giomp-overlap)
            HSA_XNACK=1 GICC_PROXY_ENABLED=1 GICC_SKIP_DWQ_INIT=1 \
            GICC_HALO_IPC=1 GICC_HALO_OVERLAP=1 \
            ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
                "${common[@]}" "${GICC_BIN}" --ngpus "${gpus}" --grid "${GRID}" \
                >"${log}" 2>&1
            ;;
        diomp)
            GASNET_BACKTRACE=1 "${common[@]}" "${DIOMP_BIN}" \
                --ngpus "${gpus}" --grid "${GRID}" >"${log}" 2>&1
            ;;
        mpi)
            MPICH_GPU_SUPPORT_ENABLED=1 \
                "${common[@]}" "${MPI_WRAP}" "${MPI_BIN}" \
                --ngpus "${gpus}" --grid "${GRID}" >"${log}" 2>&1
            ;;
        *)
            echo "unknown backend: ${backend}" >&2
            return 2
            ;;
    esac
}

echo "backend,nodes,gpus,trial,grid,max_modeling_s,rank0_modeling_s,verify" >"${OUT}/trials.csv"
for nodes in "${NODES[@]}"; do
    gpus=$((nodes * 8))
    for trial in $(seq 1 "${RUNS}"); do
        for backend in "${BACKENDS[@]}"; do
            log="${OUT}/${backend}_g${gpus}_t${trial}.log"
            ok=0
            for attempt in $(seq 1 "${MAX_TRIES}"); do
                echo "RUN backend=${backend} nodes=${nodes} gpus=${gpus} trial=${trial} attempt=${attempt}"
                if run_case "${backend}" "${nodes}" "${gpus}" "${trial}" "${log}" &&
                   ! grep -qiE 'fault|aborted|out of memory|killed' "${log}" &&
                   grep -q 'Time modeling:' "${log}"; then
                    ok=1
                    break
                fi
                echo "RETRY backend=${backend} gpus=${gpus} trial=${trial} attempt=${attempt}" >&2
            done
            if (( ! ok )); then
                echo "failed after ${MAX_TRIES} attempts: ${backend}, ${gpus} GPUs, trial ${trial}" >&2
                exit 1
            fi
            read -r maximum rank0 < <(
                awk '
                    /Time modeling:/ {
                        value=$(NF-1) + 0.0;
                        if (!seen || value > maximum) maximum=value;
                        seen=1;
                        if ($1 ~ /^0:/) rank0=value;
                    }
                    END { if (seen) printf "%.9f %.9f\n", maximum, rank0 }
                ' "${log}"
            )
            verify=PASS
            if ! grep -qE 'FINAL (field_xor|min_u)' "${log}"; then verify=FAIL; fi
            printf '%s,%d,%d,%d,%d,%.9f,%.9f,%s\n' \
                "${backend}" "${nodes}" "${gpus}" "${trial}" "${GRID}" \
                "${maximum}" "${rank0}" "${verify}" >>"${OUT}/trials.csv"
        done
    done
done

{
    echo "backend,nodes,gpus,grid,runs,median_rank0_modeling_s,min_rank0_modeling_s,max_rank0_modeling_s,median_slowest_rank_s,verify"
    for nodes in "${NODES[@]}"; do
        gpus=$((nodes * 8))
        for backend in "${BACKENDS[@]}"; do
            awk -F, -v backend="${backend}" -v nodes="${nodes}" -v gpus="${gpus}" \
                -v grid="${GRID}" -v runs="${RUNS}" '
                NR > 1 && $1 == backend && $2 == nodes {
                    slow[++n]=$6;
                    value[n]=$7;
                    if (n == 1 || $7 < minimum) minimum=$7;
                    if (n == 1 || $7 > maximum) maximum=$7;
                    if ($8 != "PASS") verify="FAIL";
                }
                END {
                    for (i=1; i<=n; ++i) {
                        for (j=i+1; j<=n; ++j) {
                            if (value[j] < value[i]) { t=value[i]; value[i]=value[j]; value[j]=t }
                            if (slow[j] < slow[i]) { t=slow[i]; slow[i]=slow[j]; slow[j]=t }
                        }
                    }
                    if (n % 2) { median=value[(n+1)/2]; slowmedian=slow[(n+1)/2]; }
                    else { median=(value[n/2]+value[n/2+1])/2.0; slowmedian=(slow[n/2]+slow[n/2+1])/2.0; }
                    if (verify == "") verify="PASS";
                    printf "%s,%d,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%s\n", \
                        backend, nodes, gpus, grid, runs, median, minimum, maximum, slowmedian, verify;
                }
            ' "${OUT}/trials.csv"
        done
    done
} >"${OUT}/results.csv"

if grep -q ',FAIL$' "${OUT}/trials.csv"; then
    echo "Minimod verification failure" >&2
    exit 1
fi
echo "MINIMOD_EVAL_PASS"
echo "TRIALS=${OUT}/trials.csv"
echo "CSV=${OUT}/results.csv"
