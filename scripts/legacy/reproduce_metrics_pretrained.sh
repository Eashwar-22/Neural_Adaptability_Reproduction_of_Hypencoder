#!/bin/bash
#SBATCH --job-name="hype_repro_metrics"
#SBATCH --partition=a100-galvani
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=120G
#SBATCH --output=logs/reproduction_metrics_%j.log

source ~/.bashrc
export HF_DATASETS_CACHE="/mnt/lustre/work/eickhoff/esx510/cache_dir"
export TMPDIR="/mnt/lustre/work/eickhoff/esx510/cache_dir"
export PYTHONUNBUFFERED=1

# --- PATHS ---
CHECKPOINT="checkpoints/hypencoder.6_layer_frozen/checkpoint-19650"
ENCODED_PATH="assets/encoded_items.docs"
NEIGHBORS_PATH="assets/hypencoder.6_layer.neighbor_graph.jsonl"
OUTPUT_DIR="logs/reproduction_metrics"
QUERY_JSONL="data/queries.dev.jsonl"
QREL_JSON="data/qrels.dev.json"

mkdir -p "$OUTPUT_DIR"

echo "Starting Evaluation..."

python -u hypencoder_cb/inference/approx_retrieve.py \
    "$CHECKPOINT" \
    "$ENCODED_PATH" \
    "$NEIGHBORS_PATH" \
    "$OUTPUT_DIR" \
    --query_jsonl="$QUERY_JSONL" \
    --qrel_json="$QREL_JSON" \
    --batch_size=1000 \
    --top_k=100 \
    --device="cuda" \
    --ncandidates=50 \
    --max_iter=16 \
    --do_eval=True

echo "Inference and Evaluation Complete. Check $OUTPUT_DIR/metrics"
