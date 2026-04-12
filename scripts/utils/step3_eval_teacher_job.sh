#!/bin/bash
#SBATCH --job-name=eval_teacher_full
#SBATCH --output=logs/slurm-%j.out
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

bash scripts/utils/step3_eval_teacher.sh
