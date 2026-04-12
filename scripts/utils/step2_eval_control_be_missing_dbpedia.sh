#!/bin/bash
# scripts/utils/step2_eval_control_be_missing_dbpedia.sh

# Environment
source ~/.bashrc
conda activate hype_env

# Flags
export HF_HOME="${HYPENCODER_CACHE:-./cache}"

# Model
MODEL_PATH="./checkpoints/control_biencoder_full"
BASE_OUT_DIR="./outputs/inference/control_biencoder_full"

echo "------------------------------------------------"
echo "Starting Control Bi-Encoder Evaluation (DBPedia Only)"
echo "Model: $MODEL_PATH"
echo "Output: $BASE_OUT_DIR"
echo "------------------------------------------------"

# --- 6. DBPedia ---
DBPEDIA_ENCODED="$BASE_OUT_DIR/dbpedia_encoded/encoded_items"
# We force encoding since the previous one failed mid-way likely
# Or maybe safely remove previous attempts if any
# rm -f "${DBPEDIA_ENCODED}.docs" 
# No, let's treat it as if we want to run it. If .docs exists it skips, but it failed so it shouldn't exist fully or correctly?
# The error said .docs.tmp exists. I removed it.
# Now I just run the encoding command.

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

echo "DBPedia Control BE Evaluation Complete."
