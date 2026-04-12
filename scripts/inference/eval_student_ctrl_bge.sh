#!/bin/bash
#SBATCH --job-name="eval_cbge"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_ctrl_bge_%j.log

source ~/.bashrc
conda activate hype_env

MODEL_PATH="./checkpoints/control_be_teacher_sota"
BASE_OUT_DIR="./outputs/inference/control_be_teacher_sota"

export HF_HOME="${HYPENCODER_CACHE:-./cache}"
export HF_DATASETS_CACHE="${HYPENCODER_CACHE:-./cache}"

echo "========================================================"
echo "Evaluating Control Bi-Encoder (BGE Teacher SOTA)"
echo "Model: $MODEL_PATH"
echo "========================================================"

# --- 1. MS MARCO & TREC DL ---
CORPUS_DATASET="msmarco-passage"
ENCODED_CORPUS_PATH="$BASE_OUT_DIR/msmarco_encoded/encoded_items"
DL19_DATASET="msmarco-passage/trec-dl-2019/judged"
DL20_DATASET="msmarco-passage/trec-dl-2020/judged"

if [ ! -d "$ENCODED_CORPUS_PATH" ]; then
    echo "[ENCODE] Encoding Full MS MARCO Corpus (8.8M passages)..."
    mkdir -p "$(dirname "$ENCODED_CORPUS_PATH")"
    python hypencoder_cb/inference/encode.py \
        --model_name_or_path "$MODEL_PATH" \
        --model_type "text_dual_encoder" \
        --ir_dataset_name "$CORPUS_DATASET" \
        --output_path "$ENCODED_CORPUS_PATH" \
        --batch_size 256 \
        --dtype "bf16"
fi

echo "[RETRIEVE] DL19"
mkdir -p "$BASE_OUT_DIR/dl19_results"
python hypencoder_cb/inference/retrieve.py --model_name_or_path "$MODEL_PATH" \
    --model_type "text_dual_encoder" \
    --encoded_item_path "$ENCODED_CORPUS_PATH" --output_dir "$BASE_OUT_DIR/dl19_results" \
    --ir_dataset_name "$DL19_DATASET" --query_max_length 64 --top_k 1000 --batch_size 4096 --dtype "bf16" --do_eval True

echo "[RETRIEVE] DL20"
mkdir -p "$BASE_OUT_DIR/dl20_results"
python hypencoder_cb/inference/retrieve.py --model_name_or_path "$MODEL_PATH" \
    --model_type "text_dual_encoder" \
    --encoded_item_path "$ENCODED_CORPUS_PATH" --output_dir "$BASE_OUT_DIR/dl20_results" \
    --ir_dataset_name "$DL20_DATASET" --query_max_length 64 --top_k 1000 --batch_size 4096 --dtype "bf16" --do_eval True

# --- 2. BEIR Datasets ---
DATASETS=(
    "covid|beir/trec-covid"
    "fiqa|beir/fiqa"
    "touche|beir/webis-touche2020/v2"
    "nfcorpus|beir/nfcorpus"
)

for ENTRY in "${DATASETS[@]}"; do
    IFS="|" read -r SHORT_NAME IR_DATASET <<< "$ENTRY"
    echo "--------------------------------------------------------"
    echo "Processing: $SHORT_NAME ($IR_DATASET)"
    
    ENCODED_PATH="$BASE_OUT_DIR/${SHORT_NAME}_encoded/encoded_items"
    RESULTS_DIR="$BASE_OUT_DIR/${SHORT_NAME}_results"
    mkdir -p "$RESULTS_DIR"
    
    if [ ! -d "${ENCODED_PATH}" ]; then
        echo "[ENCODE] Encoding..."
        mkdir -p "$(dirname "$ENCODED_PATH")"
        python hypencoder_cb/inference/encode.py \
            --model_name_or_path "$MODEL_PATH" \
            --model_type "text_dual_encoder" \
            --ir_dataset_name "$IR_DATASET" \
            --output_path "$ENCODED_PATH" \
            --batch_size 256 \
            --dtype "bf16"
    fi
    
    echo "[RETRIEVE] Retrieving..."
    python hypencoder_cb/inference/retrieve.py \
        --model_name_or_path "$MODEL_PATH" \
        --model_type "text_dual_encoder" \
        --encoded_item_path "$ENCODED_PATH" \
        --output_dir "$RESULTS_DIR" \
        --ir_dataset_name "$IR_DATASET" \
        --query_max_length 64 \
        --top_k 1000 \
        --batch_size 4096 \
        --dtype "bf16" \
        --do_eval True
done

echo "========================================================"
echo "All Evaluations Complete."
