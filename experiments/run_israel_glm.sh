#!/bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1600M
#SBATCH --time=6:00:00
#SBATCH --output=output_%j_%a.txt
#SBATCH --error=error_%j_%a.log
#SBATCH --job-name=israel_model
#SBATCH --array=1-1120

module load anaconda
conda activate pyproj

srun python israel_model_runner.py ${SLURM_ARRAY_TASK_ID}