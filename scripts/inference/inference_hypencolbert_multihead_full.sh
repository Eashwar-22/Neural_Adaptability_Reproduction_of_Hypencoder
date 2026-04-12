#!/bin/bash
#SBATCH --job-name="full_multi_inf"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/inference_full_multi_%j.log

# ==============================================================================
# SCRIPT: inference_hypencolbert_multihead_full.sh
# PURPOSE: Run Sharded Inference for FULL Multi-Head HypenColBERT (32 Heads)
# USAGE:   sbatch scripts/inference/inference_hypencolbert_multihead_full.sh <dataset_key> [checkpoint_path]
# EXAMPLE: sbatch scripts/inference/inference_hypencolbert_multihead_full.sh dl19
#          (Defaults to latest checkpoint in checkpoints/hypen_colbert_multihead_full)
# ==============================================================================

source ~/.bashrc
conda activate hype_env

# --- ARGS ---
DATASET_KEY=$1
CHECKPOINT_PATH=$2

# --- DEFAULTS ---
DEFAULT_CHECKPOINT_DIR="./checkpoints/hypen_colbert_multihead_full"

if [ -z "$DATASET_KEY" ]; then
    echo "Usage: sbatch scripts/inference/inference_hypencolbert_multihead_full.sh <dataset_key> [checkpoint_path]"
    exit 1
fi

if [ -z "$CHECKPOINT_PATH" ]; then
    # Auto-detect latest checkpoint
    if [ -d "$DEFAULT_CHECKPOINT_DIR" ]; then
        # Find directory with highest number (checkpoint-XXXX)
        LATEST_CHECKPOINT=$(find "$DEFAULT_CHECKPOINT_DIR" -maxdepth 1 -name "checkpoint-*" -type d | sort -V | tail -n 1)
        if [ -z "$LATEST_CHECKPOINT" ]; then
             echo "Error: No checkpoints found in $DEFAULT_CHECKPOINT_DIR"
             exit 1
        fi
        CHECKPOINT_PATH="$LATEST_CHECKPOINT"
        echo "Auto-detected latest checkpoint: $CHECKPOINT_PATH"
    else
        echo "Error: Default checkpoint directory $DEFAULT_CHECKPOINT_DIR does not exist."
        exit 1
    fi
fi

# --- CONFIGURATION ---
BASE_OUT_DIR="./outputs/inference/hypencolbert_multihead_full"

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

RESULTS_DIR="$BASE_OUT_DIR/${DATASET_KEY}_results"

echo "========================================================"
echo "Running FULL Multi-Head HypenColBERT Inference"
echo "Dataset: $DATASET_KEY ($IR_DATASET)"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output Dir: $RESULTS_DIR"
echo "Strategy: Sum of MaxSims (Implicit in Q-Net)"
echo "========================================================"

# --- RUN RETRIEVAL (Sharded) ---
mkdir -p "$RESULTS_DIR"

# Note: We use "ir_datasets:$IR_DATASET" as the document source string
# Doc Batch = 2000 (Fits in H100 with 128-dim + 180 tokens)
# Query Batch = 32

python hypencoder_cb/inference/retrieve_colbert.py \
    --model_name_or_path "$CHECKPOINT_PATH" \
    --document_source "ir_datasets:$IR_DATASET" \
    --ir_dataset_name "$IR_DATASET" \
    --output_dir "$RESULTS_DIR" \
    --doc_batch_size 2000 \
    --query_batch_size 32 \
    --query_max_length 32 \
    --doc_max_length 180 \
    --top_k 1000 \
    --dtype "bf16" \
    --aggregation_strategy "sum_max" \
    --do_eval True

echo "[DONE] Results saved into $RESULTS_DIR"
