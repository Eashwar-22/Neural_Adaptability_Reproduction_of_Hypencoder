#!/bin/bash

# Move to the project root directory
cd ./

echo "Current directory: $(pwd)"

# --- CONFIGURATION ---
export MODEL_NAME_OR_PATH="./checkpoints/hypencoder.6_layer_lora/checkpoint-4000"
export IR_DATASET_NAME="trec-tot/2023/dev"
# Fixed: Point to a file, not just a directory
export ENCODED_OUTPUT_PATH="./assets/trec_tot_encoded_lora_r8/encoded_items.docs"
export RETRIEVAL_DIR="./outputs/inference/trec_tot_results_lora_r8"

export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=1

# Fix: Set HOME to current dir to avoid docarray cache permission issues in some envs
export HOME=$PWD

# Fix: Create parent dir for the file
mkdir -p "$(dirname "$ENCODED_OUTPUT_PATH")"
mkdir -p "$RETRIEVAL_DIR"

echo "Model: $MODEL_NAME_OR_PATH"
echo "Dataset: $IR_DATASET_NAME"

# --- STEP 1: ENCODE ---
if [ -f "$ENCODED_OUTPUT_PATH" ]; then
    echo "Encoded file $ENCODED_OUTPUT_PATH already exists. Skipping encoding."
else
    # Remove stale tmp files just in case
    rm -f "$ENCODED_OUTPUT_PATH" "$ENCODED_OUTPUT_PATH.tmp"
    echo "--- Starting Encoding ---"
    # Fix: Remove stale tmp files
    python hypencoder_cb/inference/encode.py \
        --model_name_or_path=$MODEL_NAME_OR_PATH \
        --output_path=$ENCODED_OUTPUT_PATH \
        --ir_dataset_name=$IR_DATASET_NAME \
        --item_text_key="text" \
        --batch_size=2048 \
        --dtype="bf16"
    
    if [ $? -ne 0 ]; then
        echo "Encoding failed!"
        exit 1
    fi
fi

# --- STEP 2: RETRIEVE (Exact Search) ---
echo "--- Starting Retrieval ---"
python hypencoder_cb/inference/retrieve.py \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --encoded_item_path=$ENCODED_OUTPUT_PATH \
    --output_dir=$RETRIEVAL_DIR \
    --ir_dataset_name=$IR_DATASET_NAME \
    --query_max_length=512 \
    --top_k=1000 \
    --do_eval=True \
    --dtype="bf16" \
    --include_content=True

echo "Pipeline Complete! Check results in $RETRIEVAL_DIR"
