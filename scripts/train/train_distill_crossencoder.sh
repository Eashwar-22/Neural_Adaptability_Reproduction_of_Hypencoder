#!/bin/bash
#SBATCH --job-name="dist_ce_full"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --exclude=mlcbm005,mlcbm007
#SBATCH --output=logs/train_distill_crossencoder_%j.log

source ~/.bashrc
conda activate hype_env

CONFIG_FILE="configs/distill_crossencoder_full.yaml"

# Ensure data exists before starting (basic check)
if [ ! -f "data/triples.train.crossencoder.jsonl" ]; then
    echo "Error: Training data 'data/triples.train.crossencoder.jsonl' not found!"
    exit 1
fi

echo "Starting Cross-Encoder Distillation Training (4 GPUs)..."
echo "Config: $CONFIG_FILE"

export OMP_NUM_THREADS=8

# Run training with torchrun for distributed data parallel
torchrun --nproc_per_node=4 \
    hypencoder_cb/train/train.py \
    --config "$CONFIG_FILE"
