#!/bin/bash
#SBATCH --job-name="hype_encode"
#SBATCH --partition=a100-galvani
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=120G
#SBATCH --output=logs/step1_encode_%j.log

source ~/.bashrc
export HF_DATASETS_CACHE="/mnt/lustre/work/eickhoff/esx510/cache_dir"
export TMPDIR="/mnt/lustre/work/eickhoff/esx510/cache_dir"
export PYTHONUNBUFFERED=1

# --- PATHS ---
CHECKPOINT="checkpoints/hypencoder.6_layer_frozen/checkpoint-19650"
INPUT_DATA="data/collection.jsonl"
OUTPUT_PATH="assets/encoded_items_new.docs"

echo "--- STARTING ENCODING ---"
echo "Model:  $CHECKPOINT"
echo "Input:  $INPUT_DATA"
echo "Output: $OUTPUT_PATH"

/mnt/lustre/work/eickhoff/esx510/hype_env/bin/python -u hypencoder_cb/inference/encode.py \
    --model_name_or_path="$CHECKPOINT" \
    --jsonl_path="$INPUT_DATA" \
    --output_path="$OUTPUT_PATH" \
    --batch_size=1024 \
    --max_length=196 \
    --device="cuda" \
    --dtype="fp32"

echo "Encoding Complete."