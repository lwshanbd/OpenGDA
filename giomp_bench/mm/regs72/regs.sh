#!/bin/bash
# regs.sh BIN...: registers of the inner and PML kernels in each binary.
L=/shared/data1/Users/l1065028/llvm-22.1.7/bin
CO=/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131/Linux_aarch64/24.5/cuda/12.4/bin/cuobjdump
W=$(mktemp -d); trap 'rm -rf $W' EXIT
for b in "$@"; do
  $L/llvm-objcopy --dump-section .llvm.offloading=$W/off.bin $b /dev/null
  (cd $W && $L/llvm-offload-binary off.bin --image=file=img.o,triple=nvptx64-nvidia-cuda,arch=sm_90)
  echo "$(basename $b): $($CO --dump-resource-usage $W/img.o | grep -A1 -E 'inner_3d|pml_3d' | grep -oE '_l[0-9]+|REG:[0-9]+' | paste -sd' ')"
done
