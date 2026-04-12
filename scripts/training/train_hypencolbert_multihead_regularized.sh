#!/bin/bash
#SBATCH --job-name=reg_run
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --output=logs/reg_run_%j.out
#SBATCH --error=logs/reg_run_%j.err
#SBATCH --exclusive
#SBATCH --exclude=mlcbm015

source .bashrc
conda activate hype_env

export WANDB_MODE=offline
export OMP_NUM_THREADS=8

# Clear cache to ensure fresh start
cd $SLURM_SUBMIT_DIR

echo "Starting Regularized Training (Dropout + Shallow Q-Net)..."

torchrun --nproc_per_node=8 \
    hypencoder_cb/train/train_colbert.py \
    --config_path my_configs/hypencolbert_multihead_regularized.yaml
