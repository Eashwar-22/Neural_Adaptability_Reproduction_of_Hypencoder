#!/bin/bash
#SBATCH --job-name="hype_lora"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --exclude=mlcbm015
#SBATCH --output=logs/train_8gpu_lora_%j.log

source ~/.bashrc
conda activate hype_env

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export OMP_NUM_THREADS=8
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# --- SETUP: Define Paths ---
# 1. RAM Disk (Fast reading for raw data)
JOB_DIR="/dev/shm/hype_lora_$SLURM_JOBID"
mkdir -p $JOB_DIR

# 2. Lustre Cache (Massive storage for processing)
LUSTRE_CACHE="${HYPENCODER_CACHE:-./cache}"
mkdir -p $LUSTRE_CACHE

echo "--- Moving data to RAM Disk ($JOB_DIR) ---"

# Do NOT copy raw data to RAM (it invalidates the cache because path changes)
# cp data/triples.train.jsonl $JOB_DIR/data.jsonl

# Copy config to RAM
cp my_configs/hypencoder.6_layer_lora.yaml $JOB_DIR/config.yaml

# Update config to point to PERSISTENT data path (enables Caching)
# Using absolute path to the data file
ABS_DATA_PATH="./data/triples.train.jsonl"
sed -i "s|training_data_jsonl: .*|training_data_jsonl: $ABS_DATA_PATH|g" $JOB_DIR/config.yaml

# --- Redirect Cache to Lustre ---
echo "--- Setting HF Cache to Lustre ($LUSTRE_CACHE) ---"
export HF_HOME=$LUSTRE_CACHE
export HF_DATASETS_CACHE=$LUSTRE_CACHE
# export TMPDIR=$LUSTRE_CACHE
# export TEMP=$LUSTRE_CACHE
# export TMP=$LUSTRE_CACHE

echo "Data prep complete. Launching LoRA training..."

# Launch training
accelerate launch \
    --multi_gpu \
    --num_machines=1 \
    --mixed_precision=no \
    --num_processes=8 \
    --main_process_port=$MASTER_PORT \
    hypencoder_cb/train/train.py \
    $JOB_DIR/config.yaml

# Cleanup RAM after job finishes
rm -rf $JOB_DIR
echo "Cleanup complete."
