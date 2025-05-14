#!/bin/bash -l
#SBATCH -A p200XXX
#SBATCH -p gpu
#SBATCH --qos=default
#SBATCH -N 2
#SBATCH --gpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=06:00:00
#SBATCH --job-name=modernbert_ddp
#SBATCH --output=logs/modernbert_%j.out
#SBATCH --error=logs/modernbert_%j.err
#SBATCH --export=ALL

ulimit -n 4096

set -eo pipefail
echo "SLURM_JOB_ID      : $SLURM_JOB_ID"
echo "SLURM_NODELIST    : $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES on each task will be set by Slurm"

module load CUDA/12.6.0

source /project/scratch/p200667/miniconda/etc/profile.d/conda.sh
conda activate bert24

export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=ib0        # InfiniBand interface
export OMP_NUM_THREADS=8

CONFIG=/project/scratch/p200667/timpal0l/ModernBERT/training/modernbert-large-learning-rate-decay.yaml

srun composer main.py "$CONFIG"
