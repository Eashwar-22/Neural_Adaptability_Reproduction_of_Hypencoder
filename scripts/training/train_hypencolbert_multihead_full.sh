#!/bin/bash
#SBATCH --job-name=full_run
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --output=logs/freq_full_run_%j.out
#SBATCH --error=logs/freq_full_run_%j.err
#SBATCH --exclusive
#SBATCH --exclude=mlcbm015

source .bashrc
conda activate hype_env

export WANDB_MODE=offline
export OMP_NUM_THREADS=8

# Clear cache to ensure fresh start
cd $SLURM_SUBMIT_DIR
rm -rf .cache/huggingface/datasets

echo "Starting Full Training (250k Steps)..."

torchrun --nproc_per_node=8 \
    hypencoder_cb/train/train_colbert.py \
    --config_path my_configs/hypencolbert_multihead_full.yaml
