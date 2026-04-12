#!/bin/bash
#SBATCH --job-name="hype_base_rescue"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=logs/inference_rescue_%j.log

source ~/.bashrc
conda activate hype_env

MODEL_PATH="./checkpoints/hypencoder.6_layer_full_real_opt"
ENCODED_PATH="./outputs/inference/full_real_opt/msmarco_encoded/encoded_items"
RESULTS_DIR="./outputs/inference/full_real_opt/msmarco_results"
IR_DATASET="msmarco-passage/dev/small"

mkdir -p "$RESULTS_DIR"

echo "Rescuing Baseline Retrieval..."
echo "Model: $MODEL_PATH"
echo "Encoded: $ENCODED_PATH"

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

echo "[DONE]"
