#!/bin/bash
#SBATCH --job-name="hype_train"
#SBATCH --partition=a100-galvani
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=01:30:00
#SBATCH --output=logs/train_8gpu_%j.log

source ~/.bashrc
source /mnt/lustre/work/eickhoff/esx510/hype_env/bin/activate

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export OMP_NUM_THREADS=8

# --- SETUP: Define Paths ---
# 1. RAM Disk (Fast reading for raw data)
JOB_DIR="/dev/shm/hype_$SLURM_JOBID"
mkdir -p $JOB_DIR

# 2. Lustre Cache (Massive storage for processing)
# We move the cache HERE to fix "No space left on device"
LUSTRE_CACHE="/mnt/lustre/work/eickhoff/esx510/cache_dir"
mkdir -p $LUSTRE_CACHE

echo "--- Moving data to RAM Disk ($JOB_DIR) ---"

# Copy raw data to RAM
cp data/triples.train.tokenized.jsonl $JOB_DIR/data.jsonl

# Copy config to RAM
cp my_configs/hypencoder.6_layer_frozen.yaml $JOB_DIR/config.yaml

# Update config to point to RAM data
sed -i "s|training_data_jsonl: .*|training_data_jsonl: $JOB_DIR/data.jsonl|g" $JOB_DIR/config.yaml

# --- CRITICAL FIX: Redirect Cache to Lustre ---
echo "--- Setting HF Cache to Lustre ($LUSTRE_CACHE) ---"
export HF_HOME=$LUSTRE_CACHE
export HF_DATASETS_CACHE=$LUSTRE_CACHE
export TMPDIR=$LUSTRE_CACHE
export TEMP=$LUSTRE_CACHE
export TMP=$LUSTRE_CACHE

echo "Data prep complete. Launching training..."

# Launch training
/mnt/lustre/work/eickhoff/esx510/hype_env/bin/accelerate launch \
    --multi_gpu \
    --num_processes=8 \
    --main_process_port=$MASTER_PORT \
    hypencoder_cb/train/train.py \
    $JOB_DIR/config.yaml

# Cleanup RAM after job finishes
rm -rf $JOB_DIR
echo "Cleanup complete."