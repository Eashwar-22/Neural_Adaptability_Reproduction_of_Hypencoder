#!/bin/bash
#SBATCH --job-name="touche_r64_a256"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/inference_touche_r64_alpha256_%j.log

# Move to project root
cd ./

export HF_HOME="${HYPENCODER_CACHE:-./cache}"
export HF_DATASETS_CACHE="${HYPENCODER_CACHE:-./cache}"
export TRANSFORMERS_CACHE="${HYPENCODER_CACHE:-./cache}"
export HF_HUB_OFFLINE=0

# Fix: Set HOME to current dir to avoid docarray cache permission/path issues
export HOME=$PWD

# Use merged checkpoint associated with Alpha 256
MODEL_PATH="./checkpoints/hypencoder.6_layer_lora_r64_alpha256/merged_checkpoint-4500"
OUTPUT_DIR="outputs/inference/touche_results_lora_r64_alpha256"
# Fix: Use ABSOLUTE PATH to prevent docarray from prepending cache dir
ENCODED_OUTPUT_PATH="./assets/touche_encoded_lora_r64_alpha256/encoded_items.docs"
IR_DATASET_NAME="beir/webis-touche2020/v2"

echo "Running inference with model: $MODEL_PATH"
echo "Dataset: $IR_DATASET_NAME"
echo "Output: $OUTPUT_DIR"

# Ensure directories exist
mkdir -p "$(dirname "$ENCODED_OUTPUT_PATH")"
mkdir -p "$OUTPUT_DIR"

# --- STEP 1: ENCODE ---
echo "--- Starting Encoding ---"
python hypencoder_cb/inference/encode.py \
    --model_name_or_path "$MODEL_PATH" \
    --output_path "$ENCODED_OUTPUT_PATH" \
    --ir_dataset_name "$IR_DATASET_NAME" \
    --item_text_key="text" \
    --batch_size 256 \
    --dtype "bf16"

if [ $? -ne 0 ]; then
    echo "Encoding failed!"
    exit 1
fi

# --- STEP 2: RETRIEVE ---
echo "--- Starting Retrieval ---"
python hypencoder_cb/inference/retrieve.py \
    --model_name_or_path "$MODEL_PATH" \
    --encoded_item_path "$ENCODED_OUTPUT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --ir_dataset_name "$IR_DATASET_NAME" \
    --query_max_length 64 \
    --top_k 1000 \
    --do_eval True \
    --dtype "bf16" \
    --include_content True
