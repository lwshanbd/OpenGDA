#!/bin/bash
# Build NPB FT as OpenMP target + GPU-aware MPI and as GiOMP, all from
# ft_giomp.cpp with the same kernels. The plain builds go through
# build_giomp_example.sh (no pass plugin); `giomp_stream_pipe` goes through
# build_giomp_after.sh, which lets the z FFT kernel send its blocks itself.
#
#   mpi                FT_MPI=1  Isend/Irecv of the transpose blocks
#   mpi_alltoall       FT_MPI=2  MPI_Alltoall with pack/unpack, as in NPB-MPI
#   mpi_stream         FT_MPI=1 FT_STREAM=1
#   giomp              ompx_put of the transpose blocks
#   giomp_stream       FT_STREAM=1 (the z FFT writes the peer's half in order)
#   giomp_stream_pipe  as giomp_stream, built with the pass plugin
#
# FT_FLAGS is added to every build (e.g. "-DFT_TL=1024 -DFT_ZMAX=1024" for
# class D); compare variants only with the same FT_FLAGS.
#
# Usage: build_ft.sh [OUTDIR] [VARIANT...]   (default: build_ofi/npbft, all)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GICC_ROOT="${GICC_ROOT:-$(cd "${HERE}/../../.." && pwd)}"
OUT="${1:-${GICC_ROOT}/build_ofi/npbft}"
shift || true
VARIANTS=("$@")
[[ ${#VARIANTS[@]} -eq 0 ]] && VARIANTS=(mpi mpi_alltoall mpi_stream giomp giomp_stream giomp_stream_pipe)
mkdir -p "${OUT}"
for v in "${VARIANTS[@]}"; do
    script=build_giomp_example.sh
    lto="-foffload-lto"
    case "${v}" in
        mpi)               defs="-DFT_MPI=1" ;;
        mpi_alltoall)      defs="-DFT_MPI=2" ;;
        mpi_stream)        defs="-DFT_MPI=1 -DFT_STREAM=1" ;;
        giomp)             defs="" ;;
        giomp_stream)      defs="-DFT_STREAM=1" ;;
        giomp_stream_pipe) defs="-DFT_STREAM=1"; script=build_giomp_after.sh; lto="" ;;
        *) echo "unknown variant ${v}" >&2; exit 2 ;;
    esac
    GICC_ROOT="${GICC_ROOT}" GIOMP_EXTRA_FLAGS="${lto} ${defs} ${FT_FLAGS:-}" \
        bash "${HERE}/../${script}" "${HERE}/ft_giomp.cpp" "${OUT}/ft_${v}"
done
