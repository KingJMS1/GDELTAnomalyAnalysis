#!/bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1200M
#SBATCH --time=8:00:00
#SBATCH --output=conv_analysis_model_run_%j_%a.txt
#SBATCH --job-name=israel_model
#SBATCH --array=1-1120

module load anaconda
conda activate pyproj

srun python israel_model_runner.py ${SLURM_ARRAY_TASK_ID}