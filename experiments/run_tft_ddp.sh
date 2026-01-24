#!/bin/bash
#SBATCH --job-name=tft_ddp
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log

# Load necessary modules (adjust paths as needed)
echo "Loading"
module load anaconda
conda activate pyproj
echo "Load complete"

# Set environment variables for rendezvous
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4)) # Use a random port based on job ID
export NCCL_SOCKET_IFNAME=ib0

# Run the training script with torchrun
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    israel_panel_ddp.py