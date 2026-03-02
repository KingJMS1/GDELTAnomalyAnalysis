#!/bin/bash

#SBATCH --job-name=hybrid_p1
#SBATCH --array=0-989
#SBATCH --cpus-per-task=1       
#SBATCH --mem=3G                
#SBATCH --time=04:00:00        
#SBATCH --output=logs_hybrid/array_p1_%A_%a.out
#SBATCH --error=logs_hybrid/array_p1_%A_%a.err

module load anaconda
source activate gdelt_env

mkdir -p logs_hybrid

python tft_tsmixer_hybrid_gdelt.py