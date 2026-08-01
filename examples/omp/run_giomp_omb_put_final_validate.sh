#!/bin/bash
# Final same-node-set OMB PUT bandwidth validation for the paper.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BIN="${GIOMP_OMB_PUT_BW_BIN:-${GICC_ROOT}/build_ofi/giomp_omb_put_bw}"
OUT="${1:-${GICC_ROOT}/build_ofi/omb_put_final_validate}"
QUEUE="${GIOMP_QUEUE:-pdebug}"

if [[ "${GIOMP_OMB_PUT_FINAL_INNER:-0}" != "1" ]]; then
    exec flux alloc -q "${QUEUE}" -t 25m -N2 -n2 -c 16 -g1 \
        --cwd="${GICC_ROOT}" --env=GIOMP_OMB_PUT_FINAL_INNER=1 \
        bash "${BASH_SOURCE[0]}" "${OUT}"
fi

mkdir -p "${OUT}"
COMMON=(flux run -t 8m -N2 -n2 -c 16 -g1 -o mpibind=off
        --env=HSA_XNACK=1
        --env=GICC_HALO_IPC=0
        --env=PMI_MAX_KVS_ENTRIES=512)
BENCH=(--window=64 --min-size=4096 --max-size=4194304)

for rep in 1 2 3 4 5; do
    case "${rep}" in
        1|4) modes=(proxy gpu-trigger mpi-rma) ;;
        2|5) modes=(gpu-trigger mpi-rma proxy) ;;
        3)   modes=(mpi-rma proxy gpu-trigger) ;;
    esac
    for mode in "${modes[@]}"; do
        case "${mode}" in
            proxy)
                "${COMMON[@]}" --env=GICC_PROXY_ENABLED=1 \
                    --env=GICC_SKIP_DWQ_INIT=1 --env=GICC_HALO_DWQ=0 \
                    --env=GICC_NUM_PROXY_THREADS=1 \
                    "${BIN}" --transport=proxy --proxy-lanes=1 \
                    --in-kernel-windows "${BENCH[@]}" \
                    >"${OUT}/${mode}-r${rep}.log" 2>&1
                ;;
            gpu-trigger)
                "${COMMON[@]}" --env=GICC_HALO_DWQ=1 \
                    --env=GICC_DWQ_CQ_THREAD=0 \
                    --env=GICC_DWQ_ASYNC_STAGE=1 \
                    "${BIN}" --transport=gpu-trigger "${BENCH[@]}" \
                    >"${OUT}/${mode}-r${rep}.log" 2>&1
                ;;
            mpi-rma)
                "${COMMON[@]}" --env=MPICH_GPU_SUPPORT_ENABLED=1 \
                    --env=GICC_SKIP_DWQ_INIT=1 \
                    "${BIN}" --transport=mpi-rma "${BENCH[@]}" \
                    >"${OUT}/${mode}-r${rep}.log" 2>&1
                ;;
        esac
    done
done

for log in "${OUT}"/*.log; do
    grep -q '^ALL_CASES_PASS' "${log}"
done

echo "transport,rep,size_bytes,window,iterations,warmup,MBps,us_per_message,verify" \
    >"${OUT}/results.csv"
for rep in 1 2 3 4 5; do
    for mode in proxy gpu-trigger mpi-rma; do
        awk -F, -v m="${mode}" -v r="${rep}" \
            'BEGIN{OFS=","} $1=="RESULT" {print m,r,$3,$4,$5,$6,$7,$8,$9}' \
            "${OUT}/${mode}-r${rep}.log" >>"${OUT}/results.csv"
    done
done

echo "transport,size_bytes,window,median_MBps,median_us_per_message" \
    >"${OUT}/summary.csv"
awk -F, '
    NR > 1 {
        key = $1 SUBSEP $3
        count[key]++
        bw[key, count[key]] = $7
        us[key, count[key]] = $8
        window[key] = $4
    }
    END {
        for (key in count) {
            for (i = 1; i <= count[key]; ++i) {
                for (j = i + 1; j <= count[key]; ++j) {
                    if (us[key, i] > us[key, j]) {
                        tmp = us[key, i]
                        us[key, i] = us[key, j]
                        us[key, j] = tmp
                        tmp = bw[key, i]
                        bw[key, i] = bw[key, j]
                        bw[key, j] = tmp
                    }
                }
            }
            split(key, fields, SUBSEP)
            middle = int(count[key] / 2) + 1
            printf "%s,%s,%s,%.6f,%.6f\n", fields[1], fields[2], \
                   window[key], bw[key, middle], us[key, middle]
        }
    }
' "${OUT}/results.csv" >>"${OUT}/summary.csv"

echo "CSV: ${OUT}/results.csv"
echo "Summary: ${OUT}/summary.csv"
