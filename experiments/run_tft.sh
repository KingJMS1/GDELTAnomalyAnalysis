#!/bin/bash
#SBATCH --job-name=small_deep20
#SBATCH --output=small_deep20_%j.txt    
#SBATCH --ntasks=1                        # number of processes to be run
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1                 # every node wants one GPU
#SBATCH --constraint=h100                 # get h100s
#SBATCH --time=3:00:00

module load anaconda
conda activate pyproj

python israel_panel_tft.py