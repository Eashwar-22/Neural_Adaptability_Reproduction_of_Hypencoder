#!/bin/bash

# Move to the project root directory
cd ./

echo "Current directory: $(pwd)"

# --- CONFIGURATION ---
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
export IR_DATASET_NAME="msmarco-passage/trec-dl-hard"
# Using the shared MS MARCO encoded corpus (17GB)
export ENCODED_OUTPUT_PATH="./assets/msmarco_encoded/encoded_items.docs"
export RETRIEVAL_DIR="./outputs/inference/trec_dl_hard_results_pretrained"

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
# Skipping encoding as we use the shared MS MARCO corpus

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
    --include_content=False

echo "Pipeline Complete! Check results in $RETRIEVAL_DIR"
