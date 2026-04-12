#!/bin/bash

# Move to the project root directory
cd ./

echo "Current directory: $(pwd)"

# --- CONFIGURATION ---
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
export IR_DATASET_NAME="msmarco-passage/trec-dl-2020/judged"
# This 17GB file appears to be the MS MARCO encoded corpus
export ENCODED_OUTPUT_PATH="./assets/msmarco_encoded/encoded_items.docs"
# Existing results directory
export RETRIEVAL_DIR="./outputs/inference/trec_dl_2020_8M_a100_results_pretrained"

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
# (Commented out as inference is already done)
# echo "--- Starting Encoding ---"
# python hypencoder_cb/inference/encode.py \
#     --model_name_or_path=$MODEL_NAME_OR_PATH \
#     --output_path=$ENCODED_OUTPUT_PATH \
#     --ir_dataset_name=$IR_DATASET_NAME \
#     --item_text_key="text" \
#     --batch_size=32 \
#     --dtype="bf16"

# if [ $? -ne 0 ]; then
#     echo "Encoding failed!"
#     exit 1
# fi

# --- STEP 2: RETRIEVE (Exact Search) ---
# (Commented out as inference is already done)
# echo "--- Starting Retrieval ---"
# python hypencoder_cb/inference/retrieve.py \
#     --model_name_or_path=$MODEL_NAME_OR_PATH \
#     --encoded_item_path=$ENCODED_OUTPUT_PATH \
#     --output_dir=$RETRIEVAL_DIR \
#     --ir_dataset_name=$IR_DATASET_NAME \
#     --query_max_length=512 \
#     --top_k=1000 \
#     --do_eval=True \
#     --dtype="bf16"

echo "Pipeline Configuration Restored."
echo "Results are located in: $RETRIEVAL_DIR"
