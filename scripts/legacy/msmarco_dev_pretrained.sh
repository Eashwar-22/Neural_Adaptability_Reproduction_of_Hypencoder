#!/bin/bash

# Move to the project root directory
cd ./

echo "Current directory: $(pwd)"

# --- CONFIGURATION ---
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
export IR_DATASET_NAME="msmarco-passage/dev/small"
export ENCODED_OUTPUT_PATH="./assets/msmarco_encoded/encoded_items.docs"
export RETRIEVAL_DIR="./outputs/inference/msmarco_dev_results_pretrained"

export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=1
export HOME=$PWD

mkdir -p "$RETRIEVAL_DIR"

echo "Model: $MODEL_NAME_OR_PATH"
echo "Dataset: $IR_DATASET_NAME"
echo "Encoded Path: $ENCODED_OUTPUT_PATH"
echo "Output Dir: $RETRIEVAL_DIR"

# --- STEP 1: ENCODE ---
# Skipping encoding as 'assets/msmarco_encoded/encoded_items.docs' should already exist (shared with TREC DL).
# Uncomment below if you need to re-encode (Warning: Takes time for 8.8M docs).

# echo "--- Starting Encoding ---"
# python hypencoder_cb/inference/encode.py \
#     --model_name_or_path=$MODEL_NAME_OR_PATH \
#     --output_path=$ENCODED_OUTPUT_PATH \
#     --ir_dataset_name=$IR_DATASET_NAME \
#     --item_text_key="text" \
#     --batch_size=32 \
#     --dtype="bf16"

# --- STEP 2: RETRIEVE (Exact Search) ---
echo "--- Starting Retrieval ---"
python hypencoder_cb/inference/retrieve.py \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --encoded_item_path=$ENCODED_OUTPUT_PATH \
    --output_dir=$RETRIEVAL_DIR \
    --ir_dataset_name=$IR_DATASET_NAME \
    --query_max_length=64 \
    --top_k=1000 \
    --do_eval=True \
    --dtype="bf16"

echo "Pipeline Complete! Check results in $RETRIEVAL_DIR"
