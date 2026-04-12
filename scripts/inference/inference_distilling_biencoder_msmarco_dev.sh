#!/bin/bash
#SBATCH --job-name="bi_ms_dev"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=logs/inference_biencoder_msmarco_dev_%j.log
#SBATCH --output=logs/inference_biencoder_msmarco_dev_%j.log

source ~/.bashrc
conda activate hype_env

# --- CONFIGURATION ---
MODEL_PATH="./checkpoints/distill_biencoder_full"
BASE_OUT_DIR="./outputs/inference/distill_biencoder_full"

# Reuse the encoded corpus from the TREC DL job
ENCODED_CORPUS_PATH="$BASE_OUT_DIR/msmarco_encoded/encoded_items"
MSMARCO_DEV_DATASET="msmarco-passage/dev/judged"

export HF_HOME="${HYPENCODER_CACHE:-./cache}"
export HF_DATASETS_CACHE="${HYPENCODER_CACHE:-./cache}"

echo "========================================================"
echo "Evaluating Distilled Bi-Encoder on MS MARCO Dev"
echo "Model: $MODEL_PATH"
echo "Corpus Path: $ENCODED_CORPUS_PATH"
echo "========================================================"

# Check if encoded corpus exists
if [ ! -f "${ENCODED_CORPUS_PATH}.docs" ]; then
    echo "[ERROR] Encoded corpus not found at ${ENCODED_CORPUS_PATH}.docs"
    echo "Did the dependency job (TREC DL) fail or not run encoding?"
    exit 1
fi

echo "[RETRIEVE] MS MARCO Dev"
DEV_OUT="$BASE_OUT_DIR/msmarco_dev_results"
mkdir -p "$DEV_OUT"

python hypencoder_cb/inference/retrieve.py \
    --model_name_or_path "$MODEL_PATH" \
    --encoded_item_path "$ENCODED_CORPUS_PATH" \
    --output_dir "$DEV_OUT" \
    --ir_dataset_name "$MSMARCO_DEV_DATASET" \
    --query_max_length 64 \
    --top_k 1000 \
    --batch_size 4096 \
    --dtype "bf16" \
    --do_eval True

echo "[DONE] MS MARCO Dev Evaluation Complete."
