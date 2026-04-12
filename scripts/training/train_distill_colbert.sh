#!/bin/bash
#SBATCH --job-name=verify_colbert
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/verify_colbert_%j.out
#SBATCH --error=logs/verify_colbert_%j.err
#SBATCH --exclude=mlcbm015,mlcbm005

source .bashrc
conda activate hype_env

export WANDB_MODE=offline
export OMP_NUM_THREADS=4
export TORCH_COMPILE_DISABLE=1

cd $SLURM_SUBMIT_DIR

echo "Starting ColBERTv2 Verification Distillation..."
echo "Config: my_configs/distill_colbert_verification.yaml"

torchrun --nproc_per_node=1 \
    hypencoder_cb/train/train.py \
    --config_path my_configs/distill_colbert_verification.yaml
