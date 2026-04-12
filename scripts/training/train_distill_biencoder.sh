#!/bin/bash
#SBATCH --job-name=verify_bioenc
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/verify_biencoder_%j.out
#SBATCH --error=logs/verify_biencoder_%j.err
#SBATCH --exclude=mlcbm015

source .bashrc
conda activate hype_env

export WANDB_MODE=offline
export OMP_NUM_THREADS=4

# Clear cache to ensure fresh start
cd $SLURM_SUBMIT_DIR

echo "Starting Bi-Encoder Verification Distillation..."
echo "Config: my_configs/distill_biencoder_verification.yaml"

torchrun --nproc_per_node=1 \
    hypencoder_cb/train/train.py \
    --config_path my_configs/distill_biencoder_verification.yaml
