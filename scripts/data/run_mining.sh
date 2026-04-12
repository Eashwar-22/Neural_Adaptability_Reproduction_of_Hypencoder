#!/bin/bash
#SBATCH --job-name=hard_neg_mining
#SBATCH --output=logs/mining_%j.log
#SBATCH --partition=h100-ferranti
#SBATCH --exclude=mlcbm004
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8

source .bashrc
conda activate hype_env

export PYTHONPATH=.
cd .

python scripts/data/mine_hard_negatives.py \
    --checkpoint_path "./checkpoints/hypencoder.6_layer_full_real_opt/checkpoint-250000" \
    --output_path "./data/triples.train_hard.jsonl"
