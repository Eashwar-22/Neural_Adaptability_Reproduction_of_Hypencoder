#!/bin/bash
#SBATCH --job-name="ctrl_bge"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_control_be_bge_%j.log

source ~/.bashrc
conda activate hype_env

CONFIG_FILE="configs/train_control_be_sota.yaml"

# Ensure data exists before starting
if [ ! -f "data/triples.train.bge_v2_m3.jsonl" ]; then
    echo "Error: Training data 'data/triples.train.bge_v2_m3.jsonl' not found!"
    exit 1
fi

echo "Starting Control Bi-Encoder Training with BGE Teacher (8 GPUs)..."
echo "Config: $CONFIG_FILE"

export OMP_NUM_THREADS=8

# Run training with torchrun for distributed data parallel
torchrun --nproc_per_node=8 \
    hypencoder_cb/train/train.py \
    --config_path "$CONFIG_FILE"
