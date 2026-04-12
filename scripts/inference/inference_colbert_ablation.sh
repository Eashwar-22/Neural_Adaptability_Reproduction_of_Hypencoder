#!/bin/bash
#SBATCH --job-name="hype_col_inf"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/inference_col_%j.log

source ~/.bashrc
conda activate hype_env

# --- ARGS ---
DATASET_KEY=$1
CHECKPOINT_PATH=$2

if [ -z "$DATASET_KEY" ] || [ -z "$CHECKPOINT_PATH" ]; then
    echo "Usage: sbatch scripts/run_inference_colbert.sh <dataset_key> <checkpoint_path>"
    echo "Example: sbatch scripts/run_inference_colbert.sh msmarco checkpoints/hypencoder_colbert/checkpoint-10000"
    exit 1
fi

# --- CONFIGURATION ---
BASE_OUT_DIR="./outputs/inference/hypen_colbert"

export HF_HOME="${HYPENCODER_CACHE:-./cache}"
export HF_DATASETS_CACHE="${HYPENCODER_CACHE:-./cache}"

# Map Key to Dataset Name
case $DATASET_KEY in
  "msmarco")
    IR_DATASET="msmarco-passage/dev/small"
    ;;
  "dl19")
    IR_DATASET="msmarco-passage/trec-dl-2019/judged"
    ;;
  "dl20")
    IR_DATASET="msmarco-passage/trec-dl-2020/judged"
    ;;
  "scifact")
    IR_DATASET="beir/scifact/test"
    ;;
  "nfcorpus")
    IR_DATASET="beir/nfcorpus/test"
    ;;
  "fiqa")
    IR_DATASET="beir/fiqa/test"
    ;;
  "covid")
    IR_DATASET="beir/trec-covid"
    ;;
  "arguana")
    IR_DATASET="beir/arguana"
    ;;
  "dbpedia")
    IR_DATASET="beir/dbpedia-entity/test"
    ;;
  "touche")
    IR_DATASET="beir/webis-touche2020/v2"
    ;;
  *)
    echo "Unknown dataset key: $DATASET_KEY"
    exit 1
    ;;
esac

RESULTS_DIR="$BASE_OUT_DIR/${DATASET_KEY}_results_top5"

echo "========================================================"
echo "Running Hypen-ColBERT Inference (Ablation: Sum-Top-5)"
echo "Dataset: $DATASET_KEY ($IR_DATASET)"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output Dir: $RESULTS_DIR"
echo "========================================================"

# --- RUN RETRIEVAL (Sharded) ---
mkdir -p "$RESULTS_DIR"

# Note: We use "ir_datasets:$IR_DATASET" as the document source string
# Doc Batch = 2000 (Fits in H100 with 128-dim + 180 tokens)
# Query Batch = 100

python hypencoder_cb/inference/retrieve_colbert.py \
    --model_name_or_path "$CHECKPOINT_PATH" \
    --document_source "ir_datasets:$IR_DATASET" \
    --ir_dataset_name "$IR_DATASET" \
    --output_dir "$RESULTS_DIR" \
    --doc_batch_size 2000 \
    --query_batch_size 10 \
    --query_max_length 32 \
    --doc_max_length 180 \
    --top_k 1000 \
    --dtype "bf16" \
    --do_eval True \
    --aggregation_strategy "sum_top_5"

echo "[DONE] Results saved into $RESULTS_DIR"
