#!/bin/bash
#SBATCH --job-name=modernbert_composer
#SBATCH --account=project_462000936
#SBATCH --partition=standard-g
#SBATCH --cpus-per-task=32
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH -t 1:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

mkdir -p logs
rm -f logs/latest.out logs/latest.err
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.out logs/latest.out
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.err logs/latest.err

# auto-fail on any errors in this script
set -eo pipefail

# logging script's variables/commands for future debug needs
set -x

echo "PWD" $PWD

module purge
module load LUMI/24.03
module load partition/G                         # GPU view
module load cpeGNU/24.03                        # swaps to PrgEnv-gnu/8.5.0
module load rocm/6.0.3                          # exact ROCm you used
module use $HOME/EasyBuild/modules/all
module load aws-ofi-rccl/17d41cb-cpeGNU-24.03
module load pytorch/2.5

#Variables for distributed enviroment
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES))

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export OMP_NUM_THREADS=8

#####

# FIXED NCCL Configuration for LUMI
#export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
#export NCCL_NET_GDR_LEVEL=3

# Try different NCCL backends - comment/uncomment as needed
# Option 1: Disable OFI and use socket-based communication
export NCCL_NET="Socket"

# Option 2: If you want to try OFI with specific settings (alternative to Option 1)
# export NCCL_NET="OFI"
# export NCCL_OFI_USE_GDR=0
# export OFI_CXI_DISABLE_HOST_REGISTER=1
# export OFI_CXI_OPTIMIZED_MRS=false

# General NCCL debugging and error handling
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_ASYNC_ERROR_HANDLING=1

# ROCm specific settings for LUMI
#export HSA_FORCE_FINE_GRAIN_PCIE=1
#export NCCL_IB_DISABLE=0
#export NCCL_IB_HCA=^mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3

# PyTorch distributed settings
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
#export TORCH_NCCL_BLOCKING_WAIT=1

NODES=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
NNODES=${#NODES[@]}
HEAD_NODE=${NODES[0]}

# DDP parameters
NPROC=8

CONFIG=/scratch/project_462000936/timpal0l/ModernBERT/training/modernbert-large-learning-rate-decay.yaml

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

run_compose() {
    local NODE=$1
    local NODE_RANK=$2

    echo ">>> Launching node_rank=$NODE_RANK on $NODE"
    srun --nodelist=$NODE \
          --ntasks=1 \
          --cpus-per-task=32 \
          --gres=gpu:8 \
        composer \
          --nproc        $NPROC \
          --world_size   $WORLD_SIZE \
          --node_rank    $NODE_RANK \
          --base_rank    $(( NODE_RANK * NPROC )) \
          --master_addr  $MASTER_ADDR \
          --master_port  $MASTER_PORT \
          --stdout	 logs/modernbert_${SLURM_JOB_ID}_rank{rank}.out \
          --stderr	 logs/modernbert_${SLURM_JOB_ID}_rank{rank}.err \
          --verbose \
        main.py "$CONFIG"
}

# Launch worker nodes (ranks 1 to NNODES-1) in the background
for (( NODE_RANK=1; NODE_RANK<NNODES; NODE_RANK++ )); do
    run_compose "${NODES[$NODE_RANK]}" $NODE_RANK &
done

# Launch head node as rank 0 in the foreground
run_compose $HEAD_NODE 0

# Wait for all background launches to finish
wait
echo "All ranks finished."
