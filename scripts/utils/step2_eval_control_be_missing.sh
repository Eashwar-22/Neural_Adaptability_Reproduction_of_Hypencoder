#!/bin/bash
# scripts/utils/step2_eval_control_be_missing.sh

# Environment
source ~/.bashrc
conda activate hype_env

# Flags
export HF_HOME="${HYPENCODER_CACHE:-./cache}"

# Model
MODEL_PATH="./checkpoints/control_biencoder_full"
BASE_OUT_DIR="./outputs/inference/control_biencoder_full"

echo "------------------------------------------------"
echo "Starting Control Bi-Encoder Evaluation (Missing Datasets)"
echo "Model: $MODEL_PATH"
echo "Output: $BASE_OUT_DIR"
echo "------------------------------------------------"

# --- 5. Touché v2 ---
TOUCHE_ENCODED="$BASE_OUT_DIR/touche_encoded/encoded_items"
if [ -f "${TOUCHE_ENCODED}.docs" ]; then
    echo "[Touché v2] Found existing encoding. Skipping encoding."
else
    echo "[Touché v2] Encoding..."
    mkdir -p "$(dirname "$TOUCHE_ENCODED")"
    python hypencoder_cb/inference/encode.py \
        --model_name_or_path "$MODEL_PATH" \
        --model_type "text_dual_encoder" \
        --ir_dataset_name "beir/webis-touche2020/v2" \
        --output_path "$TOUCHE_ENCODED" \
        --batch_size 256 \
        --dtype "bf16"
fi

if [ -f "$BASE_OUT_DIR/touche_results/metrics/aggregated_metrics.txt" ]; then
    echo "[Touché v2] Found existing results. Skipping retrieval."
else
    echo "[Touché v2] Retrieval..."
    python scripts/utils/retrieve_biencoder.py \
        --model_name_or_path "$MODEL_PATH" \
        --encoded_item_path "${TOUCHE_ENCODED}.docs" \
        --output_dir "$BASE_OUT_DIR/touche_results" \
        --ir_dataset_name "beir/webis-touche2020/v2" \
        --query_max_length 64 \
        --top_k 1000 \
        --batch_size 128 \
        --dtype "bf16" \
        --do_eval True
fi

# --- 6. DBPedia ---
DBPEDIA_ENCODED="$BASE_OUT_DIR/dbpedia_encoded/encoded_items"
if [ -f "${DBPEDIA_ENCODED}.docs" ]; then
    echo "[DBPedia] Found existing encoding. Skipping encoding."
else
    echo "[DBPedia] Encoding..."
    mkdir -p "$(dirname "$DBPEDIA_ENCODED")"
    python hypencoder_cb/inference/encode.py \
        --model_name_or_path "$MODEL_PATH" \
        --model_type "text_dual_encoder" \
        --ir_dataset_name "beir/dbpedia-entity/test" \
        --output_path "$DBPEDIA_ENCODED" \
        --batch_size 256 \
        --dtype "bf16"
fi

if [ -f "$BASE_OUT_DIR/dbpedia_results/metrics/aggregated_metrics.txt" ]; then
    echo "[DBPedia] Found existing results. Skipping retrieval."
else
    echo "[DBPedia] Retrieval..."
    python scripts/utils/retrieve_biencoder.py \
        --model_name_or_path "$MODEL_PATH" \
        --encoded_item_path "${DBPEDIA_ENCODED}.docs" \
        --output_dir "$BASE_OUT_DIR/dbpedia_results" \
        --ir_dataset_name "beir/dbpedia-entity/test" \
        --query_max_length 64 \
        --top_k 1000 \
        --batch_size 128 \
        --dtype "bf16" \
        --do_eval True
fi

echo "Missing Control BE Evaluations Complete."
