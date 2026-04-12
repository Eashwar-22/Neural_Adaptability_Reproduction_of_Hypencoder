#!/bin/bash
#SBATCH --job-name="bi_beir"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/inference_biencoder_beir_%j.log

source ~/.bashrc
conda activate hype_env

# --- CONFIGURATION ---
MODEL_PATH="./checkpoints/distill_biencoder_full"
BASE_OUT_DIR="./outputs/inference/distill_biencoder_full"

export HF_HOME="${HYPENCODER_CACHE:-./cache}"
export HF_DATASETS_CACHE="${HYPENCODER_CACHE:-./cache}"

# Define datasets to eval
# Format: "SHORT_NAME|IR_DATASET_ID"
DATASETS=(
    "covid|beir/trec-covid"
    "fiqa|beir/fiqa"
    "touche|beir/webis-touche2020/v2"
)

echo "========================================================"
echo "Evaluating Distilled Bi-Encoder on BEIR (Small)"
echo "Model: $MODEL_PATH"
echo "========================================================"

for ENTRY in "${DATASETS[@]}"; do
    IFS="|" read -r SHORT_NAME IR_DATASET <<< "$ENTRY"
    
    echo "--------------------------------------------------------"
    echo "Processing: $SHORT_NAME ($IR_DATASET)"
    
    ENCODED_PATH="$BASE_OUT_DIR/${SHORT_NAME}_encoded/encoded_items"
    RESULTS_DIR="$BASE_OUT_DIR/${SHORT_NAME}_results"
    
    mkdir -p "$RESULTS_DIR"
    
    # 1. ENCODE
    # Remove old encoding to be safe (small datasets) or check existence
    if [ -d "${ENCODED_PATH}" ]; then
        echo "[ENCODE] Exists at $ENCODED_PATH. Skipping."
    else
        echo "[ENCODE] Encoding..."
        mkdir -p "$(dirname "$ENCODED_PATH")"
        python hypencoder_cb/inference/encode.py \
            --model_name_or_path "$MODEL_PATH" \
            --ir_dataset_name "$IR_DATASET" \
            --output_path "$ENCODED_PATH" \
            --batch_size 256 \
            --dtype "bf16"
            
        if [ $? -ne 0 ]; then
            echo "[ERROR] Encoding failed for $SHORT_NAME"
            continue
        fi
    fi
    
    # 2. RETRIEVE
    echo "[RETRIEVE] Retrieving..."
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
        
    echo "[DONE] $SHORT_NAME"
done

echo "========================================================"
echo "All BEIR tasks complete."
