#!/bin/bash
#SBATCH --job-name="interactive_umap"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/interactive_umap_%j.log

source ~/.bashrc
conda activate hype_env

echo "Running Interactive UMAP analysis..."
python scripts/analysis/interactive_umap.py \
    --model_path checkpoints/hypencoder.6_layer_full_real_opt \
    --data_path data/triples.train.bge.jsonl \
    --output_dir docs/analysis/interactive_umap \
    --num_queries 1000

echo "Done."
