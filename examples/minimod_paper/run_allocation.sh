#!/bin/bash
# Run all arms for one randomized, allocation-level paired replicate.
set -euo pipefail

if test "$#" -ne 8; then
    echo "usage: $0 KIND NODES RPN REP GRID STEPS ORDER OUTDIR" >&2
    exit 2
fi

KIND="$1"
NODES="$2"
RPN="$3"
REP="$4"
GRID="$5"
STEPS="$6"
ORDER="$7"
OUTDIR="$8"

# flux batch copies this script into a private job directory, so dirname($0)
# is not the repository.  submit.sh sets --cwd to the repository root.
ROOT="${GICC_ROOT:-$PWD}"
BIN_DIR="$ROOT/build_ofi/minimod_paper/bin"
RANKS=$((NODES * RPN))
mkdir -p "$OUTDIR"

case "$KIND" in
    standard)
        test -x "$BIN_DIR/minimod_default"
        test -x "$BIN_DIR/minimod_trigger"
        test -x "$BIN_DIR/minimod_proxy"
        ;;
    o5)
        test -x "$BIN_DIR/minimod_o5"
        ;;
    *) echo "unknown KIND=$KIND" >&2; exit 2 ;;
esac

printf 'ALLOC kind=%s nodes=%s rpn=%s ranks=%s rep=%s grid=%s steps=%s order=%s job=%s\n' \
    "$KIND" "$NODES" "$RPN" "$RANKS" "$REP" "$GRID" "$STEPS" \
    "$ORDER" "${FLUX_JOB_ID:-unknown}"

IFS=, read -r -a ARMS <<< "$ORDER"
for arm in "${ARMS[@]}"; do
    binary=""
    dwq_mode=1
    overlap=0
    o5_mode=""
    case "$arm" in
        default_serial)    binary=minimod_default ;;
        default_overlap)   binary=minimod_default; overlap=1 ;;
        trigger_serial)    binary=minimod_trigger ;;
        trigger_overlap)   binary=minimod_trigger; overlap=1 ;;
        proxy_serial)      binary=minimod_proxy; dwq_mode=0 ;;
        proxy_overlap)     binary=minimod_proxy; dwq_mode=0; overlap=1 ;;
        mirrored_serial)   binary=minimod_o5; o5_mode=mirrored ;;
        unmirrored_serial) binary=minimod_o5; o5_mode=unmirrored ;;
        *) echo "unknown arm=$arm" >&2; exit 2 ;;
    esac

    log="$OUTDIR/${KIND}-n${NODES}-rpn${RPN}-rep${REP}-${arm}.log"
    printf 'BEGIN arm=%s log=%s time=%s\n' "$arm" "$log" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'RUN kind=%s arm=%s nodes=%s rpn=%s ranks=%s rep=%s grid=%s steps=%s binary=%s\n' \
        "$KIND" "$arm" "$NODES" "$RPN" "$RANKS" "$REP" "$GRID" \
        "$STEPS" "$BIN_DIR/$binary" > "$log"

    env -u GICC_SKIP_DWQ_INIT -u MPICH_GPU_SUPPORT_ENABLED \
        FI_MR_CACHE_MAX_COUNT=0 PMI_MAX_KVS_ENTRIES=512 \
        GICC_PROXY_ENABLED=1 GICC_NUM_PROXY_THREADS=1 \
        GICC_DWQ_MODE="$dwq_mode" GICC_MINIMOD_OVERLAP="$overlap" \
        GICC_MINIMOD_O5_MODE="$o5_mode" \
        timeout 75s flux run -N "$NODES" -n "$RANKS" \
            -c 8 -g 1 \
            "$BIN_DIR/$binary" --ngpus "$RANKS" --grid "$GRID" \
            --nsteps "$STEPS" < /dev/null >> "$log" 2>&1

    # Fail in the allocation, rather than letting a truncated log become a
    # timing sample.  Detailed checksum and route validation is repeated by
    # analyze.py after all allocation-level replicates complete.
    test "$(grep -c '^CHECKSUM rank ' "$log")" -eq "$RANKS"
    test "$(grep -c '^GICC_ROUTE rank ' "$log")" -eq "$RANKS"
    test "$(grep -c 'Time kernel:' "$log")" -eq "$RANKS"
    printf 'END arm=%s time=%s\n' "$arm" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

printf 'ALLOC_DONE kind=%s nodes=%s rpn=%s rep=%s\n' \
    "$KIND" "$NODES" "$RPN" "$REP"
