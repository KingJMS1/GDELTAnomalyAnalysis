#!/bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1600M
#SBATCH --time=12:00:00
#SBATCH --output=logs_shrinkage/output_%j_%a.txt
#SBATCH --error=logs_shrinkage/error_%j_%a.log
#SBATCH --job-name=shrinkage_model
#SBATCH --array=1-990

module load anaconda
conda activate gdelt_env

srun python shrinkage_model_runner.py ${SLURM_ARRAY_TASK_ID}