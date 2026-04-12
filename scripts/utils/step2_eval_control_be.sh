#!/bin/bash
# scripts/utils/step2_eval_control_be.sh

# Environment
source ~/.bashrc
conda activate hype_env

# Flags
export HF_HOME="${HYPENCODER_CACHE:-./cache}"

# Model
MODEL_PATH="./checkpoints/control_biencoder_full"
BASE_OUT_DIR="./outputs/inference/control_biencoder_full"

echo "------------------------------------------------"
echo "Starting Control Bi-Encoder Evaluation"
echo "Model: $MODEL_PATH"
echo "Output: $BASE_OUT_DIR"
echo "------------------------------------------------"

# --- 1. MS MARCO & TREC DL ---
# We encode MS MARCO Passage corpus once (used by both DL19 and DL20)
MSMARCO_ENCODED="$BASE_OUT_DIR/msmarco_encoded/encoded_items"
MSMARCO_DATASET="msmarco-passage" 

if [ -f "${MSMARCO_ENCODED}.docs" ]; then
    echo "[MS MARCO] Found existing encoding at ${MSMARCO_ENCODED}.docs. Skipping."
else
    echo "[MS MARCO] Encoding Corpus..."
    mkdir -p "$(dirname "$MSMARCO_ENCODED")"
    python hypencoder_cb/inference/encode.py \
        --model_name_or_path "$MODEL_PATH" \
        --model_type "text_dual_encoder" \
        --ir_dataset_name "msmarco-passage" \
        --output_path "$MSMARCO_ENCODED" \
        --batch_size 256 \
        --dtype "bf16"
fi

# TREC DL 19 Inference
echo "[TREC DL 19] Running Retrieval..."
python scripts/utils/retrieve_biencoder.py \
    --model_name_or_path "$MODEL_PATH" \
    --encoded_item_path "${MSMARCO_ENCODED}.docs" \
    --output_dir "$BASE_OUT_DIR/dl19_results" \
    --ir_dataset_name "msmarco-passage/trec-dl-2019/judged" \
    --query_max_length 64 \
    --top_k 1000 \
    --batch_size 128 \
    --dtype "bf16" \
    --do_eval True

# TREC DL 20 Inference
echo "[TREC DL 20] Running Retrieval..."
python scripts/utils/retrieve_biencoder.py \
    --model_name_or_path "$MODEL_PATH" \
    --encoded_item_path "${MSMARCO_ENCODED}.docs" \
    --output_dir "$BASE_OUT_DIR/dl20_results" \
    --ir_dataset_name "msmarco-passage/trec-dl-2020/judged" \
    --query_max_length 64 \
    --top_k 1000 \
    --batch_size 128 \
    --dtype "bf16" \
    --do_eval True


# --- 2. TREC-COVID ---
COVID_ENCODED="$BASE_OUT_DIR/trec_covid_encoded/encoded_items"
echo "[TREC-COVID] Encoding..."
mkdir -p "$(dirname "$COVID_ENCODED")"
python hypencoder_cb/inference/encode.py \
    --model_name_or_path "$MODEL_PATH" \
    --model_type "text_dual_encoder" \
    --ir_dataset_name "beir/trec-covid" \
    --output_path "$COVID_ENCODED" \
    --batch_size 256 \
    --dtype "bf16"

echo "[TREC-COVID] Retrieval..."
python scripts/utils/retrieve_biencoder.py \
    --model_name_or_path "$MODEL_PATH" \
    --encoded_item_path "${COVID_ENCODED}.docs" \
    --output_dir "$BASE_OUT_DIR/covid_results" \
    --ir_dataset_name "beir/trec-covid" \
    --query_max_length 64 \
    --top_k 1000 \
    --batch_size 128 \
    --dtype "bf16" \
    --do_eval True


# --- 3. NFCorpus ---
NFC_ENCODED="$BASE_OUT_DIR/nfcorpus_encoded/encoded_items"
echo "[NFCorpus] Encoding..."
mkdir -p "$(dirname "$NFC_ENCODED")"
python hypencoder_cb/inference/encode.py \
    --model_name_or_path "$MODEL_PATH" \
    --model_type "text_dual_encoder" \
    --ir_dataset_name "beir/nfcorpus/test" \
    --output_path "$NFC_ENCODED" \
    --batch_size 256 \
    --dtype "bf16"

echo "[NFCorpus] Retrieval..."
python scripts/utils/retrieve_biencoder.py \
    --model_name_or_path "$MODEL_PATH" \
    --encoded_item_path "${NFC_ENCODED}.docs" \
    --output_dir "$BASE_OUT_DIR/nfcorpus_results" \
    --ir_dataset_name "beir/nfcorpus/test" \
    --query_max_length 64 \
    --top_k 1000 \
    --batch_size 128 \
    --dtype "bf16" \
    --do_eval True


# --- 4. FiQA ---
FIQA_ENCODED="$BASE_OUT_DIR/fiqa_encoded/encoded_items"
echo "[FiQA] Encoding..."
mkdir -p "$(dirname "$FIQA_ENCODED")"
python hypencoder_cb/inference/encode.py \
    --model_name_or_path "$MODEL_PATH" \
    --model_type "text_dual_encoder" \
    --ir_dataset_name "beir/fiqa/test" \
    --output_path "$FIQA_ENCODED" \
    --batch_size 256 \
    --dtype "bf16"

echo "[FiQA] Retrieval..."
python scripts/utils/retrieve_biencoder.py \
    --model_name_or_path "$MODEL_PATH" \
    --encoded_item_path "${FIQA_ENCODED}.docs" \
    --output_dir "$BASE_OUT_DIR/fiqa_results" \
    --ir_dataset_name "beir/fiqa/test" \
    --query_max_length 64 \
    --top_k 1000 \
    --batch_size 128 \
    --dtype "bf16" \
    --do_eval True


# --- 5. Touché v2 ---
TOUCHE_ENCODED="$BASE_OUT_DIR/touche_encoded/encoded_items"
echo "[Touché v2] Encoding..."
mkdir -p "$(dirname "$TOUCHE_ENCODED")"
python hypencoder_cb/inference/encode.py \
    --model_name_or_path "$MODEL_PATH" \
    --model_type "text_dual_encoder" \
    --ir_dataset_name "beir/webis-touche2020/v2" \
    --output_path "$TOUCHE_ENCODED" \
    --batch_size 256 \
    --dtype "bf16"

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


# --- 6. DBPedia ---
DBPEDIA_ENCODED="$BASE_OUT_DIR/dbpedia_encoded/encoded_items"
echo "[DBPedia] Encoding..."
mkdir -p "$(dirname "$DBPEDIA_ENCODED")"
python hypencoder_cb/inference/encode.py \
    --model_name_or_path "$MODEL_PATH" \
    --model_type "text_dual_encoder" \
    --ir_dataset_name "beir/dbpedia-entity/test" \
    --output_path "$DBPEDIA_ENCODED" \
    --batch_size 256 \
    --dtype "bf16"

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

echo "All Control BE Evaluations Complete."
