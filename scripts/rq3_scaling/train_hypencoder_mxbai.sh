#!/bin/bash
#SBATCH --job-name="hype_mxbai"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_hypencoder_mxbai_%j.log

source ~/.bashrc
conda activate hype_env

CONFIG_FILE="configs/scaling/train_hypencoder_mxbai.yaml"

# Ensure data exists before starting
if [ ! -f "data/triples.train.mxbai.jsonl" ]; then
    echo "Error: Training data 'data/triples.train.mxbai.jsonl' not found!"
    exit 1
fi

echo "Starting Hypencoder Training with MXBAI Teacher (8 GPUs)..."
echo "Config: $CONFIG_FILE"

export OMP_NUM_THREADS=8

# Run training with torchrun for distributed data parallel
torchrun --nproc_per_node=8 \
    hypencoder_cb/train/train.py \
    --config_path "$CONFIG_FILE"
