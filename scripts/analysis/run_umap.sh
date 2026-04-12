#!/bin/bash
#SBATCH --job-name="qnet_umap"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/umap_qnet_%j.log

source ~/.bashrc
conda activate hype_env

echo "Running UMAP analysis on GPU..."
python scripts/analysis/umap_qnet_layers.py \
    --model_path checkpoints/hypencoder.6_layer_full_real_opt \
    --data_path data/triples.train.bge.jsonl \
    --output_dir msc_thesis/images/umap_layers \
    --num_queries 1000

echo "Done."
