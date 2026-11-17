source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY OMPX_HEAP_SIZE=4G
cd /shared/data1/Users/l1065028/giomp_bench/moe
srun -p maple -A app --gres=gpu:1 --mem=0 --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 2 -N2 -n2 "$@" 2>&1 | grep -E "median|phases|rror"
