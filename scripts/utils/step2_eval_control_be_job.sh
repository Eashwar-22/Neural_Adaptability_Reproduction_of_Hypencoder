#!/bin/bash
#SBATCH --job-name=eval_control_be_full
#SBATCH --output=logs/slurm-%j.out
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

bash scripts/utils/step2_eval_control_be.sh
