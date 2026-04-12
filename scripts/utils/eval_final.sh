#!/bin/bash
#SBATCH --job-name="hype_eval"
#SBATCH --partition=a100-galvani
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=120G
#SBATCH --output=logs/eval_final_%j.log

source ~/.bashrc
# source /mnt/lustre/work/eickhoff/esx510/hype_env/bin/activate

# --- CRITICAL FIXES ---
export HF_DATASETS_CACHE="/mnt/lustre/work/eickhoff/esx510/cache_dir"
export TMPDIR="/mnt/lustre/work/eickhoff/esx510/cache_dir"
export PYTHONUNBUFFERED=1  # <--- Forces unbuffered output

# --- PATH CONFIGURATION ---
PROJECT_ROOT=$(pwd)
CHECKPOINT="checkpoints/hypencoder.6_layer_frozen/checkpoint-19650"
ENCODED_PATH="$PROJECT_ROOT/assets/encoded_items.docs"
NEIGHBORS_PATH="$PROJECT_ROOT/assets/hypencoder.6_layer.neighbor_graph.jsonl"
OUTPUT_DIR="logs/inference_results"

mkdir -p "$OUTPUT_DIR"

echo "--- STARTING INFERENCE ---"
echo "Checkpoint: $CHECKPOINT"
echo "Embeddings: $ENCODED_PATH"
echo "Neighbors:  $NEIGHBORS_PATH"

# --- RUN INFERENCE ---
# Added '-u' to python command
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