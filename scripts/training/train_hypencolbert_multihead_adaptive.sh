#!/bin/bash
#SBATCH --job-name=adaptive_train
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --output=logs/adaptive_train_%j.out
#SBATCH --error=logs/adaptive_train_%j.err
#SBATCH --exclusive
#SBATCH --exclude=mlcbm015

source .bashrc
conda activate hype_env

export WANDB_MODE=offline
export OMP_NUM_THREADS=8

# Clear cache to ensure fresh start
cd $SLURM_SUBMIT_DIR
rm -rf .cache/huggingface/datasets

echo "Starting Adaptive Training (250k Steps)..."
echo "Data: data/triples.train.adaptive_hard.jsonl"

torchrun --nproc_per_node=8 \
    hypencoder_cb/train/train_colbert.py \
    --config_path configs/hypencolbert_multihead_adaptive.yaml
