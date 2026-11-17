#!/bin/bash
#SBATCH -p maple
#SBATCH -A app
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH -t 40
bash /shared/data1/Users/l1065028/giomp_bench/mm/run_ov.sh $SLURM_NNODES
