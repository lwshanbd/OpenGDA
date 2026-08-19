#!/bin/bash
# Build all paper candidates from an isolated copy of Minimod.
set -euo pipefail

ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
SOURCE="${MINIMOD_SOURCE:-$ROOT/benchmarks/Minimod_MPI}"
OUT="${MINIMOD_PAPER_OUT:-$ROOT/build_ofi/minimod_paper}"
SCRIPT="$ROOT/examples/minimod_paper"
PASSES="$ROOT/tools/gicc-passes/build/libgicc-passes.so"

test -f "$SOURCE/targets/hip_gicc_unified/target_3d.cpp"
test -f "$PASSES"
mkdir -p "$OUT/bin" "$OUT/build" "$OUT/meta"

WORK=$(mktemp -d "$OUT/work.XXXXXX")
cp -a "$SOURCE/." "$WORK/"
printf '%s\n' "$WORK" > "$OUT/last_work.txt"

MAKE_ARGS=(all TARGET=hip_gicc_unified COMPILER=hipcc_gicc_unified
           GICC_USE_LTO_PASS=1 GICC_ROOT="$ROOT" GICC_PASSES_SO="$PASSES")
EXE="$WORK/main_hip_gicc_unified_hipcc_gicc_unified"

clean_build() {
    make -C "$WORK" TARGET=hip_gicc_unified COMPILER=hipcc_gicc_unified \
        GICC_USE_LTO_PASS=1 GICC_ROOT="$ROOT" clean >/dev/null
}

build_standard() {
    python3 "$SCRIPT/prepare_sources.py" --work "$WORK" --mode standard

    local meta="$OUT/meta/standard-features"
    mkdir -p "$meta"
    find "$meta" -mindepth 1 -maxdepth 1 -type f -delete
    clean_build
    env CCACHE_DISABLE=1 GICC_MODE=feature-extract GICC_META_DIR="$meta" \
        GICC_FEATURES_OUT="$meta/features.json" GICC_PROXY_ENABLED=1 \
        timeout 900 make -C "$WORK" "${MAKE_ARGS[@]}" \
        >"$OUT/build/standard-features.log" 2>&1
    test -s "$meta/features.json"

    local name hint hint_args
    for name in default trigger proxy; do
        case "$name" in
            default) hint="" ;;
            trigger) hint="$SCRIPT/hints/trigger.json" ;;
            proxy)   hint="$SCRIPT/hints/proxy.json" ;;
        esac
        meta="$OUT/meta/standard-$name"
        mkdir -p "$meta"
        find "$meta" -mindepth 1 -maxdepth 1 -type f -delete
        clean_build
        hint_args=()
        if test -n "$hint"; then hint_args=(GICC_HINT_IN="$hint"); fi
        env CCACHE_DISABLE=1 GICC_MODE=lower GICC_META_DIR="$meta" \
            GICC_PROXY_ENABLED=1 "${hint_args[@]}" \
            timeout 900 make -C "$WORK" "${MAKE_ARGS[@]}" \
            >"$OUT/build/standard-$name.log" 2>&1
        test -x "$EXE"
        cp "$EXE" "$OUT/bin/minimod_$name"
    done
}

build_o5() {
    # Restore pristine files inside the private copy, then apply the O5 rewrite.
    cp "$SOURCE/targets/hip_gicc_unified/target_3d.cpp" \
       "$WORK/targets/hip_gicc_unified/target_3d.cpp"
    cp "$SOURCE/targets/hip_gicc_unified/data_setup.cpp" \
       "$WORK/targets/hip_gicc_unified/data_setup.cpp"
    python3 "$SCRIPT/prepare_sources.py" --work "$WORK" --mode o5

    local meta="$OUT/meta/o5-features"
    mkdir -p "$meta"
    find "$meta" -mindepth 1 -maxdepth 1 -type f -delete
    clean_build
    env CCACHE_DISABLE=1 GICC_MODE=feature-extract GICC_META_DIR="$meta" \
        GICC_FEATURES_OUT="$meta/features.json" GICC_PROXY_ENABLED=1 \
        timeout 900 make -C "$WORK" "${MAKE_ARGS[@]}" \
        >"$OUT/build/o5-features.log" 2>&1
    test -s "$meta/features.json"

    python3 "$SCRIPT/make_o5_hint.py" --features "$meta/features.json" \
        --output "$OUT/meta/o5-hint.json" >"$OUT/build/o5-hint.log"

    meta="$OUT/meta/o5-lower"
    mkdir -p "$meta"
    find "$meta" -mindepth 1 -maxdepth 1 -type f -delete
    clean_build
    env CCACHE_DISABLE=1 GICC_MODE=lower GICC_META_DIR="$meta" \
        GICC_HINT_IN="$OUT/meta/o5-hint.json" GICC_PROXY_ENABLED=1 \
        timeout 900 make -C "$WORK" "${MAKE_ARGS[@]}" \
        >"$OUT/build/o5-lower.log" 2>&1
    test -x "$EXE"
    cp "$EXE" "$OUT/bin/minimod_o5"
}

case "${BUILD_SCOPE:-all}" in
    all)
        build_standard
        build_o5
        ;;
    standard)
        build_standard
        ;;
    o5)
        build_o5
        ;;
    *)
        printf 'unknown BUILD_SCOPE=%s (use all, standard, or o5)\n' \
            "${BUILD_SCOPE:-}" >&2
        exit 2
        ;;
esac

sha256sum "$OUT"/bin/minimod_* | sort > "$OUT/binary-sha256.txt"
sha256sum "$SOURCE/targets/hip_gicc_unified/target_3d.cpp" \
          "$SOURCE/targets/hip_gicc_unified/data_setup.cpp" \
          > "$OUT/source-sha256.txt"
printf 'built candidates in %s\n' "$OUT/bin"
cat "$OUT/binary-sha256.txt"
