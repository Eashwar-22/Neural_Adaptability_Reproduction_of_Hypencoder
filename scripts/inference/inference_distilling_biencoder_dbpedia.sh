#!/bin/bash
#SBATCH --job-name="bi_dbpedia"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/inference_biencoder_dbpedia_%j.log

source ~/.bashrc
conda activate hype_env

# --- CONFIGURATION ---
MODEL_PATH="./checkpoints/distill_biencoder_full"
BASE_OUT_DIR="./outputs/inference/distill_biencoder_full"

# DBPedia Config
DATASET_NAME="beir/dbpedia-entity/test"
ENCODED_PATH="$BASE_OUT_DIR/dbpedia_encoded/encoded_items"
RESULTS_DIR="$BASE_OUT_DIR/dbpedia_results"

export HF_HOME="${HYPENCODER_CACHE:-./cache}"
export HF_DATASETS_CACHE="${HYPENCODER_CACHE:-./cache}"

echo "========================================================"
echo "Evaluating Distilled Bi-Encoder on DBPedia"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_NAME"
echo "========================================================"

mkdir -p "$RESULTS_DIR"

# 1. ENCODE (~4.6M docs)
if [ -d "$ENCODED_PATH" ]; then
    echo "[ENCODE] Found existing encoded corpus at $ENCODED_PATH. Skipping."
else
    echo "[ENCODE] Encoding 4.6M documents..."
    mkdir -p "$(dirname "$ENCODED_PATH")"
    python hypencoder_cb/inference/encode.py \
        --model_name_or_path "$MODEL_PATH" \
        --ir_dataset_name "$DATASET_NAME" \
        --output_path "$ENCODED_PATH" \
        --batch_size 256 \
        --dtype "bf16"
        
    if [ $? -ne 0 ]; then
        echo "[ERROR] Encoding failed."
        exit 1
    fi
fi

# 2. RETRIEVE (~400 queries)
echo "[RETRIEVE] Retrieving..."
python hypencoder_cb/inference/retrieve.py \
    --model_name_or_path "$MODEL_PATH" \
    --encoded_item_path "$ENCODED_PATH" \
    --output_dir "$RESULTS_DIR" \
    --ir_dataset_name "$DATASET_NAME" \
    --query_max_length 64 \
    --top_k 1000 \
    --batch_size 4096 \
    --dtype "bf16" \
    --do_eval True

echo "[DONE] DBPedia Evaluation Complete."
