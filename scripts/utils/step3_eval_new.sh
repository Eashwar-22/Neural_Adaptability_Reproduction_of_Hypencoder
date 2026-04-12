#!/bin/bash
#SBATCH --job-name="hype_eval_new"
#SBATCH --partition=a100-galvani
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=120G
#SBATCH --output=logs/step3_eval_%j.log

source ~/.bashrc
export HF_DATASETS_CACHE="/mnt/lustre/work/eickhoff/esx510/cache_dir"
export TMPDIR="/mnt/lustre/work/eickhoff/esx510/cache_dir"
export PYTHONUNBUFFERED=1

# --- PATHS ---
CHECKPOINT="checkpoints/hypencoder.6_layer_frozen/checkpoint-19650"
ENCODED_PATH="assets/encoded_items_new.docs"
NEIGHBORS_PATH="assets/hypencoder_neighbors_new.jsonl"
OUTPUT_DIR="logs/inference_results_new"

mkdir -p "$OUTPUT_DIR"

/mnt/lustre/work/eickhoff/esx510/hype_env/bin/python -u hypencoder_cb/inference/approx_retrieve.py \
    "$CHECKPOINT" \
    "$ENCODED_PATH" \
    "$NEIGHBORS_PATH" \
    "$OUTPUT_DIR" \
    --query_jsonl="data/queries.dev.jsonl" \
    --batch_size=1000 \
    --top_k=100 \
    --device="cuda" \
    --ncandidates=50 \
    --max_iter=16 \
    --do_eval=False 

echo "Inference Complete."