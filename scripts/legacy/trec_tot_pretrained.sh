#!/bin/bash

# Move to the project root directory
cd ./

echo "Current directory: $(pwd)"

# --- CONFIGURATION ---
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
export IR_DATASET_NAME="trec-tot/2023/dev"
# TREC TOT has its own corpus, so we must encode it separately
export ENCODED_OUTPUT_PATH="./assets/trec_tot_encoded/encoded_items.docs"
export RETRIEVAL_DIR="./outputs/inference/trec_tot_results_pretrained"

export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=1
export HOME=$PWD

mkdir -p "$(dirname "$ENCODED_OUTPUT_PATH")"
mkdir -p "$RETRIEVAL_DIR"

echo "Model: $MODEL_NAME_OR_PATH"
echo "Dataset: $IR_DATASET_NAME"
echo "Encoded Path: $ENCODED_OUTPUT_PATH"
echo "Output Dir: $RETRIEVAL_DIR"

# --- STEP 1: ENCODE ---
echo "--- Starting Encoding ---"
# Note: TREC TOT corpus is small (~231k docs), so this should be fast.
python hypencoder_cb/inference/encode.py \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --output_path=$ENCODED_OUTPUT_PATH \
    --ir_dataset_name=$IR_DATASET_NAME \
    --item_text_key="text" \
    --batch_size=32 \
    --dtype="bf16"

if [ $? -ne 0 ]; then
    echo "Encoding failed!"
    exit 1
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
    --dtype="bf16"

echo "Pipeline Complete! Check results in $RETRIEVAL_DIR"
