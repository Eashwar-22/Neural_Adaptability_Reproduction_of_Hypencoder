#!/bin/bash
#SBATCH --job-name="hype_opt"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --exclude=mlcbm015
#SBATCH --output=logs/train_8gpu_full_real_opt_%j.log

source ~/.bashrc
conda activate hype_env

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export OMP_NUM_THREADS=8
# OPTIMIZATION: Enabled Torch Compile
export TORCH_COMPILE_DISABLE=0
export TORCHDYNAMO_DISABLE=0

# --- SETUP: Define Paths ---
# 1. RAM Disk (Fast reading for raw data)
JOB_DIR="/dev/shm/hype_opt_$SLURM_JOBID"
mkdir -p $JOB_DIR

# 2. Lustre Cache (Massive storage for processing)
LUSTRE_CACHE="${HYPENCODER_CACHE:-./cache}"
mkdir -p $LUSTRE_CACHE

# 3. Explicitly DELETE Stale Caches to be safe (though we already did, good practice)
# rm -rf data/*.dataset

echo "--- Moving data to RAM Disk ($JOB_DIR) ---"

# Copy config to RAM
cp my_configs/hypencoder_retrained.yaml $JOB_DIR/config.yaml

# Update config to point to PERSISTENT data path
ABS_DATA_PATH="./data/triples.train.jsonl"
sed -i "s|training_data_jsonl: .*|training_data_jsonl: $ABS_DATA_PATH|g" $JOB_DIR/config.yaml

echo "--- Setting HF Cache to Lustre ($LUSTRE_CACHE) ---"
export HF_HOME=$LUSTRE_CACHE
export HF_DATASETS_CACHE=$LUSTRE_CACHE

echo "Data prep complete. Launching OPTIMIZED Full Scale Training..."
echo "Config: Batch 128/GPU (Global 1024), Steps 125k, Torch Compile ON"

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
