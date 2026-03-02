#!/bin/bash
#SBATCH --job-name=sim_comp
#SBATCH --output=sim_comp_%A_%a.out
#SBATCH --error=sim_comp_%A_%a.err
#SBATCH --array=0-6
#SBATCH --cpus-per-task=1       
#SBATCH --mem=3G                
#SBATCH --time=04:00:00  

module load anaconda
source activate gdelt_env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_force_host_platform_device_count=1"

python run_sim_task.py $SLURM_ARRAY_TASK_ID