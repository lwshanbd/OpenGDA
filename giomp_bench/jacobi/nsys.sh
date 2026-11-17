#!/bin/bash
# nsys.sh N OUT BINARY [args]: rank 0 of an N-node run under nsys, kernel summary.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
N=$1; out=$2; shift 2; NSYS=/hrtc/apps/cuda/12.4.131/aarch64/rocky9/bin/nsys
export NSYS OUT=$out
srun -p maple --account=app -N $N --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=4 --mpi=pmix -t 10 \
  bash -c 'if [ "$SLURM_PROCID" = 0 ]; then exec $NSYS profile -t cuda -f true -o "$OUT" "$@"; else exec "$@"; fi' _ "$@" 2>&1 | grep -E "phases|GPUs:"
$NSYS stats -q --report cuda_gpu_kern_sum --format csv "$out.nsys-rep" 2>/dev/null | cut -d, -f2-9 | head -6
