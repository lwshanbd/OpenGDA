#!/usr/bin/env bash
# Harness for the three GICC-pass toy-kernel tests (Block M2-I).
# Compiles each .cpp to .ll with clang, runs the gicc-analysis pass via opt,
# greps the stderr for expected feature/decision tuples.
#
# Usage:  run_pass_tests.sh  <build_dir>  <clang>  <opt>
#   build_dir: where libgicc_pass.so lives
#   clang:    /opt/rocm-6.4.0/lib/llvm/bin/clang++
#   opt:      /opt/rocm-6.4.0/lib/llvm/bin/opt
#
# Exit codes: 0 all pass; 1 at least one failed; 2 harness error.

set -eo pipefail

BUILD="${1:?build dir required}"
CLANG="${2:?clang path required}"
OPT="${3:?opt path required}"

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY="${SELF_DIR}/../../policies/policy_tioga.json"
PLUGIN="${BUILD}/tools/gicc_pass/libgicc_pass.so"

[[ -f "$POLICY" ]] || { echo "missing policy: $POLICY" >&2; exit 2; }
[[ -f "$PLUGIN" ]] || { echo "missing plugin: $PLUGIN" >&2; exit 2; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

run_site() {
    local src="$1"
    local rank="$2"
    "$CLANG" -std=c++17 -O0 -emit-llvm -S -g "$src" -o "$TMP/$(basename $src .cpp).ll" 2>/dev/null
    GICC_POLICY_FILE="$POLICY" GICC_EXPLAIN=1 GICC_RANK_HINT="$rank" \
        "$OPT" -load-pass-plugin="$PLUGIN" -passes=gicc-analysis \
               "$TMP/$(basename $src .cpp).ll" -S -o /dev/null 2>&1
}

FAIL=0

check() {
    local name="$1"; shift
    local out="$1"; shift
    for want in "$@"; do
        if ! echo "$out" | grep -qE "$want"; then
            echo "FAIL [$name]: missing expected pattern: $want"
            echo "----- actual output -----"
            echo "$out"
            echo "-------------------------"
            FAIL=$((FAIL + 1))
            return
        fi
    done
    echo "PASS [$name]"
}

# Test 1 ---------------------------------------------------------------------
OUT="$(run_site "$SELF_DIR/test_all_static.cpp" 32)"
check "test_all_static" "$OUT" \
    "gicc::put.*peer=k_const" \
    "gicc::put.*size=large_const" \
    "rule_id=20" \
    "path=ofi_proxy.*pool_size=32"

# Test 2 ---------------------------------------------------------------------
OUT="$(run_site "$SELF_DIR/test_peer_dynamic.cpp" 32)"
check "test_peer_dynamic" "$OUT" \
    "gicc::put.*peer=dynamic" \
    "gicc::put.*size=large_const" \
    "rule_id=10" \
    "path=ofi_triggered.*slot_depth=4"

# Test 3 ---------------------------------------------------------------------
OUT="$(run_site "$SELF_DIR/test_all_dynamic.cpp" 32)"
check "test_all_dynamic" "$OUT" \
    "gicc::put.*peer=dynamic" \
    "gicc::put.*size=dynamic" \
    "gicc-pass: processed 1 GICC call site"

if (( FAIL > 0 )); then
    echo "$FAIL test(s) failed"
    exit 1
fi
echo "all 3 gicc-pass toy-kernel tests passed"
