#!/bin/bash
#SBATCH --job-name="bi_dl_eval"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00 # Increased time for full 8.8M corpus encoding
#SBATCH --output=logs/inference_biencoder_trec_%j.log

source ~/.bashrc
conda activate hype_env

# --- CONFIGURATION ---
MODEL_PATH="./checkpoints/distill_biencoder_full"
BASE_OUT_DIR="./outputs/inference/distill_biencoder_full"

# Common Config
CORPUS_DATASET="msmarco-passage"
ENCODED_CORPUS_PATH="$BASE_OUT_DIR/msmarco_encoded/encoded_items"
DL19_DATASET="msmarco-passage/trec-dl-2019/judged"
DL20_DATASET="msmarco-passage/trec-dl-2020/judged"

export HF_HOME="${HYPENCODER_CACHE:-./cache}"
export HF_DATASETS_CACHE="${HYPENCODER_CACHE:-./cache}"

echo "========================================================"
echo "Evaluating Distilled Bi-Encoder on TREC DL '19 & '20"
echo "Model: $MODEL_PATH"
echo "Corpus: $CORPUS_DATASET"
echo "Output Base: $BASE_OUT_DIR"
echo "========================================================"

# --- 1. ENCODE CORPUS (MS MARCO 8.8M) ---
# Check if already encoded (unlikely for this new model, so almost always runs)
if [ ! -d "$ENCODED_CORPUS_PATH" ]; then
    echo "[ENCODE] Encoding Full MS MARCO Corpus (8.8M passages)..."
    mkdir -p "$(dirname "$ENCODED_CORPUS_PATH")"
    
    python hypencoder_cb/inference/encode.py \
        --model_name_or_path "$MODEL_PATH" \
        --ir_dataset_name "$CORPUS_DATASET" \
        --output_path "$ENCODED_CORPUS_PATH" \
        --batch_size 256 \
        --dtype "bf16"
        
    if [ $? -ne 0 ]; then
        echo "[ERROR] Encoding failed."
        exit 1
    fi
else
    echo "[ENCODE] Found existing encoded corpus at $ENCODED_CORPUS_PATH. Skipping encoding."
fi

# --- 2. RETRIEVE TREC DL '19 ---
echo "--------------------------------------------------------"
echo "[RETRIEVE] TREC DL 2019"
DL19_OUT="$BASE_OUT_DIR/dl19_results"
mkdir -p "$DL19_OUT"

python hypencoder_cb/inference/retrieve.py \
    --model_name_or_path "$MODEL_PATH" \
    --encoded_item_path "$ENCODED_CORPUS_PATH" \
    --output_dir "$DL19_OUT" \
    --ir_dataset_name "$DL19_DATASET" \
    --query_max_length 64 \
    --top_k 1000 \
    --batch_size 4096 \
    --dtype "bf16" \
    --do_eval True

# --- 3. RETRIEVE TREC DL '20 ---
echo "--------------------------------------------------------"
echo "[RETRIEVE] TREC DL 2020"
DL20_OUT="$BASE_OUT_DIR/dl20_results"
mkdir -p "$DL20_OUT"

python hypencoder_cb/inference/retrieve.py \
    --model_name_or_path "$MODEL_PATH" \
    --encoded_item_path "$ENCODED_CORPUS_PATH" \
    --output_dir "$DL20_OUT" \
    --ir_dataset_name "$DL20_DATASET" \
    --query_max_length 64 \
    --top_k 1000 \
    --batch_size 4096 \
    --dtype "bf16" \
    --do_eval True

echo "========================================================"
echo "[DONE] Evaluation Complete."
