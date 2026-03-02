#!/bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1600M
#SBATCH --time=8:00:00
#SBATCH --output=logs_step2/output_%j_%a.txt
#SBATCH --error=logs_step2/error_%j_%a.log
#SBATCH --job-name=shrinkage_model_step2
#SBATCH --array=1-130

module load anaconda
source activate gdelt_env

OFFSET=990
REAL_ID=$((SLURM_ARRAY_TASK_ID + OFFSET))

srun python shrinkage_model_step2_runner.py ${REAL_ID}