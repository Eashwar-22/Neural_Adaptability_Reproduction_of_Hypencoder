#!/bin/bash
#SBATCH --job-name="hype_split_conv"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=04:00:00
#SBATCH --output=logs/convert_%j.log

source ~/.bashrc
conda activate hype_env

# 1. Split the massive file
echo "Splitting 90GB file into 3GB chunks..."
mkdir -p data/split_chunks
rm -f data/split_chunks/*

# split -C 3000M splits by bytes but respects line boundaries (perfect for jsonl)
split -d -C 3000M data/triples.train.jsonl data/split_chunks/chunk_

echo "Splitting complete. Files:"
ls -lh data/split_chunks | head

# 2. Run Parallel Conversion
echo "Starting Parallel Conversion on chunks..."
python scripts/utils/convert_data_parallel.py \
    --input_dir "data/split_chunks" \
    --output_dir "data/triples.train.jsonl.dataset" \
    --num_workers 30

echo "Conversion Complete."
# Cleanup chunks
rm -rf data/split_chunks
