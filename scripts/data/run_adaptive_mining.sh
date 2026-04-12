#!/bin/bash
#SBATCH --job-name=adaptive_mining_4gpu
#SBATCH --output=logs/adaptive_mining_4gpu_%j.log
#SBATCH --partition=h100-ferranti
#SBATCH --exclude=mlcbm004
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8

source .bashrc
conda activate hype_env

export PYTHONPATH=.
cd .

echo "Starting Adaptive Hard Negative Mining (4 GPUs)..."
echo "Checkpoint: checkpoints/hypencoder.6_layer_full_real_opt/checkpoint-250000"
echo "Output: data/triples.train.adaptive_hard.jsonl"

python scripts/data/mine_hard_negatives.py \
    --checkpoint_path "./checkpoints/hypencoder.6_layer_full_real_opt/checkpoint-250000" \
    --output_path "./data/triples.train.adaptive_hard.jsonl"
