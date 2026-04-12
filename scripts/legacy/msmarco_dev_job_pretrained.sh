#!/bin/bash
#SBATCH --job-name="hype_marcoms"
#SBATCH --partition=a100-galvani
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --output=outputs/logs/inference_%j.log

# --- SETUP ENVIRONMENT ---
source ~/.bashrc
source /mnt/lustre/work/eickhoff/esx510/hype_env/bin/activate

# --- CRITICAL FIX: PREVENT STORAGE CRASH ---
export HF_DATASETS_CACHE="/mnt/lustre/work/eickhoff/esx510/cache_dir"
export TMPDIR="/mnt/lustre/work/eickhoff/esx510/cache_dir"
mkdir -p "$TMPDIR"

# Move to project root
cd ./
echo "Running on host: $(hostname)"

# --- CONFIGURATION ---
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
export IR_DATASET_NAME="msmarco-passage/dev/small"
export ENCODED_OUTPUT_PATH="./assets/msmarco_encoded/encoded_items.docs"
export RETRIEVAL_DIR="./outputs/inference/msmarco_dev_results_pretrained"

export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=1

mkdir -p "$RETRIEVAL_DIR"
# Ensure the logs directory exists for SLURM output
mkdir -p outputs/logs

# --- STEP 2: RETRIEVE ---
echo "--- Starting Retrieval ---"
python hypencoder_cb/inference/retrieve.py \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --encoded_item_path=$ENCODED_OUTPUT_PATH \
    --output_dir=$RETRIEVAL_DIR \
    --ir_dataset_name=$IR_DATASET_NAME \
    --query_max_length=64 \
    --top_k=1000 \
    --do_eval=True \
    --dtype="bf16"

echo "Pipeline Complete! Check results in $RETRIEVAL_DIR"
