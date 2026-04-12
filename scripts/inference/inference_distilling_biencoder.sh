#!/bin/bash
#SBATCH --job-name="bi_infer_nf"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/inference_biencoder_%j.log

source ~/.bashrc
conda activate hype_env

# --- CONFIGURATION ---
# Pointing to the specific checkpoint from the bi-encoder distillation experiment
MODEL_PATH="./checkpoints/distill_biencoder_full"
BASE_OUT_DIR="./outputs/inference/distill_biencoder_full"

export HF_HOME="${HYPENCODER_CACHE:-./cache}"
export HF_DATASETS_CACHE="${HYPENCODER_CACHE:-./cache}"

# Dataset: NFCorpus (Small & Fast)
DATASET_KEY="nfcorpus"
IR_DATASET="beir/nfcorpus"
SHORT_NAME="nfcorpus"

ENCODED_PATH="$BASE_OUT_DIR/${SHORT_NAME}_encoded/encoded_items"
RESULTS_DIR="$BASE_OUT_DIR/${DATASET_KEY}_results"

echo "========================================================"
echo "Running Inference for Key: $DATASET_KEY"
echo "Dataset: $IR_DATASET"
echo "Model: $MODEL_PATH"
echo "Output Dir: $RESULTS_DIR"
echo "========================================================"

# --- 1. ENCODE ---
# Always re-encode since this is a different model than the main one
if [ -d "${ENCODED_PATH}" ]; then
    echo "[ENCODE] Found existing encoding at ${ENCODED_PATH}, removing to ensure fresh encoding..."
    rm -rf "${ENCODED_PATH}"
fi

echo "[ENCODE] Encoding corpus..."
mkdir -p "$(dirname "$ENCODED_PATH")"
python hypencoder_cb/inference/encode.py \
    --model_name_or_path "$MODEL_PATH" \
    --ir_dataset_name "$IR_DATASET" \
    --output_path "$ENCODED_PATH" \
    --batch_size 256 \
    --dtype "bf16"

if [ $? -ne 0 ]; then
    echo "[ERROR] Encoding failed."
    exit 1
fi

# --- 2. RETRIEVE ---
echo "[RETRIEVE] Running retrieval..."
mkdir -p "$RESULTS_DIR"

python hypencoder_cb/inference/retrieve.py \
    --model_name_or_path "$MODEL_PATH" \
    --encoded_item_path "$ENCODED_PATH" \
    --output_dir "$RESULTS_DIR" \
    --ir_dataset_name "$IR_DATASET" \
    --query_max_length 64 \
    --top_k 1000 \
    --batch_size 4096 \
    --dtype "bf16" \
    --do_eval True

echo "[DONE] Results saved to $RESULTS_DIR"
