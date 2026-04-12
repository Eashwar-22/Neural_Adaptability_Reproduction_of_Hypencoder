#!/bin/bash
#SBATCH --job-name="dist_bi_full"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=48:00:00 # Increased time for 50 epochs
#SBATCH --exclude=mlcbm005,mlcbm007
#SBATCH --output=logs/train_distill_biencoder_%j.log

export NCCL_TIMEOUT=3600

source ~/.bashrc
conda activate hype_env

CONFIG_FILE="configs/distill_biencoder_full.yaml"

# Ensure data exists before starting (basic check)
if [ ! -f "data/triples.train.bge.jsonl" ]; then
    echo "Error: Training data 'data/triples.train.bge.jsonl' not found!"
    exit 1
fi

echo "Starting Bi-Encoder Distillation Training (8 GPUs)..."
echo "Config: $CONFIG_FILE"

export OMP_NUM_THREADS=8

# Run training with torchrun for distributed data parallel
torchrun --nproc_per_node=8 \
    hypencoder_cb/train/train.py \
    --config_path "$CONFIG_FILE"

