#!/bin/bash
#SBATCH --job-name="hype_index"
#SBATCH --partition=a100-galvani
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=120G
#SBATCH --output=logs/step2_index_%j.log

source ~/.bashrc
export HF_DATASETS_CACHE="/mnt/lustre/work/eickhoff/esx510/cache_dir"
export TMPDIR="/mnt/lustre/work/eickhoff/esx510/cache_dir"
export PYTHONUNBUFFERED=1

# --- PATHS ---
INPUT_EMBEDDINGS="assets/encoded_items_new.docs"
OUTPUT_GRAPH="assets/hypencoder_neighbors_new.jsonl"

echo "--- STARTING INDEXING ---"
echo "Input:  $INPUT_EMBEDDINGS"
echo "Output: $OUTPUT_GRAPH"

/mnt/lustre/work/eickhoff/esx510/hype_env/bin/python -u hypencoder_cb/inference/neighbor_graph.py \
    --encoded_items_path="$INPUT_EMBEDDINGS" \
    --output_path="$OUTPUT_GRAPH" \
    --batch_size=1000 \
    --top_k=100 \
    --device="cuda" \
    --dtype="fp32" \
    --distance="l2"

echo "Indexing Complete."