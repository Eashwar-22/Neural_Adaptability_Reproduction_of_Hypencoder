#!/bin/bash
#SBATCH --job-name="touche_fv2"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/inference_touche_full_v2_%j.log

source ~/.bashrc
conda activate hype_env

export HOME=$PWD
export HF_HOME="${HYPENCODER_CACHE:-./cache}"
export HF_DATASETS_CACHE="${HYPENCODER_CACHE:-./cache}"
export TRANSFORMERS_CACHE="${HYPENCODER_CACHE:-./cache}"

# Model Path (The "Other Model" trained on 1% data)
MODEL_PATH="./checkpoints/hypencoder.6_layer_full_v2"
IR_DATASET_NAME="beir/webis-touche2020/v2"
# Use Absolute Path to avoid DocArray cache confusion
ENCODED_OUTPUT_PATH="./outputs/inference/touche_encoded_full_v2"
RETRIEVAL_DIR="./outputs/inference/touche_results_full_v2"

# Note: Do NOT mkdir ENCODED_OUTPUT_PATH as it is a file prefix for DocArray
mkdir -p $(dirname $ENCODED_OUTPUT_PATH)
mkdir -p $RETRIEVAL_DIR

echo "--- Starting Encoding ---"
# Check for the DocArray file (it appends .docs usually, but we check prefix)
if [ -f "${ENCODED_OUTPUT_PATH}.docs" ] || [ -f "${ENCODED_OUTPUT_PATH}" ]; then
    echo "Encoded items found, skipping encoding."
else
    python hypencoder_cb/inference/encode.py \
        --model_name_or_path $MODEL_PATH \
        --ir_dataset_name $IR_DATASET_NAME \
        --output_path $ENCODED_OUTPUT_PATH \
        --batch_size 256 \
        --dtype "bf16"
fi

echo "--- Starting Retrieval ---"
python hypencoder_cb/inference/retrieve.py \
    --model_name_or_path $MODEL_PATH \
    --encoded_item_path $ENCODED_OUTPUT_PATH \
    --output_dir $RETRIEVAL_DIR \
    --ir_dataset_name $IR_DATASET_NAME \
    --query_max_length 64 \
    --top_k 1000 \
    --batch_size 256 \
    --dtype "bf16" \
    --do_eval True
