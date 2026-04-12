#!/bin/bash
#SBATCH --job-name="debug_test"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=logs/debug_%j.log

echo "Hello from SLURM"
nvidia-smi
