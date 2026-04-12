#!/bin/bash
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

echo "--- MEASURING TREC-COVID ---"
export ENCODED_OUTPUT_PATH="./assets/trec_covid_encoded/encoded_items.docs"
export RETRIEVAL_DIR="./outputs/inference/trec_covid_results"
export IR_DATASET_NAME="beir/trec-covid"

/mnt/lustre/work/eickhoff/esx510/hype_env/bin/python hypencoder_cb/inference/retrieve.py \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --encoded_item_path=$ENCODED_OUTPUT_PATH \
    --output_dir=$RETRIEVAL_DIR \
    --ir_dataset_name=$IR_DATASET_NAME \
    --query_max_length=512 \
    --do_eval=False \
    --track_time=True

echo "--- MEASURING FIQA ---"
export ENCODED_OUTPUT_PATH="./assets/fiqa_encoded/encoded_items.docs"
export RETRIEVAL_DIR="./outputs/inference/fiqa_results"
export IR_DATASET_NAME="beir/fiqa/test"

/mnt/lustre/work/eickhoff/esx510/hype_env/bin/python hypencoder_cb/inference/retrieve.py \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --encoded_item_path=$ENCODED_OUTPUT_PATH \
    --output_dir=$RETRIEVAL_DIR \
    --ir_dataset_name=$IR_DATASET_NAME \
    --query_max_length=512 \
    --do_eval=False \
    --track_time=True
