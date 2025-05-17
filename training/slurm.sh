#!/bin/bash -l
#SBATCH -A p200XXX
#SBATCH -p gpu
#SBATCH --qos=default
#SBATCH --nodes=2                # total nodes
#SBATCH --ntasks-per-node=1      # one Slurm task per node
#SBATCH --gres=gpu:4             # 4 GPUs per node
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --job-name=modernbert_ddp
#SBATCH --output=logs/modernbert_%j.out
#SBATCH --error=logs/modernbert_%j.err
#SBATCH --exclusive              # no other jobs on these nodes
#SBATCH --wait-all-nodes=1       # start only once all nodes are up

set -eo pipefail
ulimit -n 8192

module purge
module load NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0

# Ensure we use the system NCCL
export LD_LIBRARY_PATH=$EBROOTNCCL/lib:$LD_LIBRARY_PATH

source /project/scratch/p200667/miniconda/etc/profile.d/conda.sh
conda activate bert2026

# Logging / debug
export LOGLEVEL=INFO
export NCCL_DEBUG=TRACE
export TORCH_CPP_LG_LEVEL=INFO

# NCCL / IB-verbs tunables
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export OMP_NUM_THREADS=8

# Gather node list and compute world‐size
NODES=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
NNODES=${#NODES[@]}
HEAD_NODE=${NODES[0]}

# Derive the IP of the head node on ib0
MASTER_ADDR=$( srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
  hostname --ip-address )
MASTER_PORT=$(( ( RANDOM % 10000 ) + 20000 ))  # random port in [20000–29999]

# DDP parameters
NPROC=4
WORLD_SIZE=$(( NNODES * NPROC ))

CONFIG=/project/scratch/p200667/timpal0l/ModernBERT/training/modernbert-large-learning-rate-decay.yaml

echo "=== DDP CONFIGURATION ==="
echo "  NODES:           ${NODES[*]}"
echo "  NNODES:          $NNODES"
echo "  HEAD_NODE:       $HEAD_NODE"
echo "  MASTER_ADDR:     $MASTER_ADDR"
echo "  MASTER_PORT:     $MASTER_PORT"
echo "  NPROC per node:  $NPROC"
echo "  WORLD_SIZE:      $WORLD_SIZE"
echo "  CONFIG:          $CONFIG"
echo "========================="

# Function to launch Composer on a given node‐rank
run_compose() {
  local NODE=$1 NODE_RANK=$2
  echo ">>> Launching node_rank=$NODE_RANK on $NODE"
  srun --nodelist=$NODE --ntasks=1 --cpus-per-task=8 \
       --gres=gpu:4 \
    composer \
      --nproc     $NPROC \
      --world_size $WORLD_SIZE \
      --node_rank  $NODE_RANK \
      --base_rank  $(( NODE_RANK * NPROC )) \
      --master_addr $MASTER_ADDR \
      --master_port $MASTER_PORT \
      --stdout    logs/modernbert_%j_rank%r.out \
      --stderr    logs/modernbert_%j_rank%r.err \
      --verbose \
    main.py "$CONFIG"
}

# Launch worker nodes (ranks 1 … NNODES-1) in background
for (( NODE_RANK=1; NODE_RANK<NNODES; NODE_RANK++ )); do
  run_compose "${NODES[$NODE_RANK]}" $NODE_RANK &
done

# Launch head node as rank 0 in foreground
run_compose $HEAD_NODE 0

# Wait for all background launches
wait
echo "All ranks finished."

