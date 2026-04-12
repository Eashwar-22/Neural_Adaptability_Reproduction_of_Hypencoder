#!/bin/bash
#SBATCH --job-name="hype_infer"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --exclude=mlcbm005
#SBATCH --output=logs/inference_%j.log

source ~/.bashrc
conda activate hype_env

# --- ARGS ---
DATASET_KEY=$1

if [ -z "$DATASET_KEY" ]; then
    echo "Usage: sbatch scripts/run_inference_opt.sh <dataset_key>"
    echo "Available keys: msmarco, dl19, dl20, touche, covid, nfcorpus, fiqa, dbpedia"
    exit 1
fi

# --- CONFIGURATION ---
MODEL_PATH="./checkpoints/hypencoder_retrained"
BASE_OUT_DIR="./outputs/inference/hypencoder_retrained"

export HF_HOME="${HYPENCODER_CACHE:-./cache}"
export HF_DATASETS_CACHE="${HYPENCODER_CACHE:-./cache}"

# Map Key to Dataset Name & Output dir
case $DATASET_KEY in
  "msmarco")
    IR_DATASET="msmarco-passage/dev/small"
    SHORT_NAME="msmarco"
    ;;
  "dl19")
    IR_DATASET="msmarco-passage/trec-dl-2019/judged"
    SHORT_NAME="msmarco"  # Shares encoding with msmarco
    ;;
  "dl20")
    IR_DATASET="msmarco-passage/trec-dl-2020/judged"
    SHORT_NAME="msmarco"  # Shares encoding with msmarco
    ;;
  "touche")
    IR_DATASET="beir/webis-touche2020/v2"
    SHORT_NAME="touche2020"
    ;;
  "covid")
    IR_DATASET="beir/trec-covid"
    SHORT_NAME="trec_covid"
    ;;
  "nfcorpus")
    IR_DATASET="beir/nfcorpus"
    SHORT_NAME="nfcorpus"
    ;;
  "fiqa")
    IR_DATASET="beir/fiqa"
    SHORT_NAME="fiqa"
    ;;
  "dbpedia")
    IR_DATASET="beir/dbpedia-entity"
    SHORT_NAME="dbpedia"
    ;;
  "dlhard")
    IR_DATASET="msmarco-passage/trec-dl-hard"
    SHORT_NAME="msmarco" # Uses same corpus as msmarco
    ;;
  "tot")
    IR_DATASET="trec-tot/2023/dev"
    SHORT_NAME="trec_tot"
    ;;
  *)
    echo "Unknown dataset key: $DATASET_KEY"
    exit 1
    ;;
esac

ENCODED_PATH="$BASE_OUT_DIR/${SHORT_NAME}_encoded/encoded_items"
RESULTS_DIR="$BASE_OUT_DIR/${DATASET_KEY}_results"

echo "========================================================"
echo "Running Inference for Key: $DATASET_KEY"
echo "Dataset: $IR_DATASET"
echo "Encoding Path: $ENCODED_PATH"
echo "Output Dir: $RESULTS_DIR"
echo "========================================================"

# --- 1. ENCODE (If needed) ---
# Check if docarray index exists (folder or file with .docs suffix)
if [ -d "${ENCODED_PATH}" ] || [ -f "${ENCODED_PATH}" ] || [ -f "${ENCODED_PATH}.docs" ]; then
    echo "[ENCODE] Index found at ${ENCODED_PATH}, skipping encoding."
else
    echo "[ENCODE] Encoding corpus..."
    mkdir -p $(dirname $ENCODED_PATH)
    python hypencoder_cb/inference/encode.py \
        --model_name_or_path "$MODEL_PATH" \
        --ir_dataset_name "$IR_DATASET" \
        --output_path "$ENCODED_PATH" \
        --batch_size 256 \
        --dtype "bf16"
        
    if [ $? -ne 0 ]; then
        echo "[ERROR] Encoding failed."
        exit 1
    fi
fi

# --- 2. RETRIEVE ---
echo "[RETRIEVE] Running retrieval..."
mkdir -p "$RESULTS_DIR"

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

echo "[DONE] Results saved to $RESULTS_DIR"
