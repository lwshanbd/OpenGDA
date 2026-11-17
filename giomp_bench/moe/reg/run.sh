source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
S="srun -p maple -A app --gres=gpu:1 --mem=0 --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 2 -N2 -n2"
for b in /shared/data1/Users/l1065028/giomp_bench/jacobi/jl5 /shared/data1/Users/l1065028/giomp_bench/moe/reg/new/jacobi_giomp_pipe; do
  echo "== $b"; $S $b -nx 16384 -ny 16384 -niter 1000 2>&1 | grep -vE "^ *[0-9]+, *[0-9.]+$" | tail -6; done
