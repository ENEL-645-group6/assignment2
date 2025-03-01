#!/bin/bash
#SBATCH --job-name=g6-a2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu


# Load conda environment
source ~/software/init-conda
conda activate pytorch

# Run the training script
python group6_assignment2_final.py

# Deactivate conda environment
conda deactivate

