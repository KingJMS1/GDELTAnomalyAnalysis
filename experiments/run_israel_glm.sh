#!/bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1200M
#SBATCH --time=1:00:00
#SBATCH --output=output_%j_%a.txt
#SBATCH --error=error_%j.log
#SBATCH --job-name=israel_model
#SBATCH --array=1-5

module load anaconda
conda activate pyproj

srun python israel_model_runner.py ${SLURM_ARRAY_TASK_ID}