#!/bin/bash
#SBATCH --job-name="hype_mh"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_8gpu_multihead_%j.log

source ~/.bashrc
conda activate hype_env

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export OMP_NUM_THREADS=8
# OPTIMIZATION: Enabled Torch Compile
export TORCH_COMPILE_DISABLE=0
export TORCHDYNAMO_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- SETUP: Define Paths ---
# 1. RAM Disk (Fast reading for raw data)
JOB_DIR="/dev/shm/hype_mh_$SLURM_JOBID"
mkdir -p $JOB_DIR

# 2. Lustre Cache (Massive storage for processing)
LUSTRE_CACHE="${HYPENCODER_CACHE:-./cache}"
mkdir -p $LUSTRE_CACHE

echo "--- Moving data to RAM Disk ($JOB_DIR) ---"

# Copy config to RAM
# CHANGED: Use multi-head config
cp my_configs/hypencolbert_multihead.yaml $JOB_DIR/config.yaml

# Update config to point to PERSISTENT data path
ABS_DATA_PATH="./data/triples.train.jsonl"
sed -i "s|training_data_jsonl: .*|training_data_jsonl: $ABS_DATA_PATH|g" $JOB_DIR/config.yaml

echo "--- Setting HF Cache to Lustre ($LUSTRE_CACHE) ---"
export HF_HOME=$LUSTRE_CACHE
export HF_DATASETS_CACHE=$LUSTRE_CACHE

echo "Data prep complete. Launching HypenColBERT Multi-Head Training..."
echo "Architecture: Multi-Head Q-Net + SumMax Aggregation"
echo "Config: hypencolbert_multihead.yaml"

# Launch training using the NEW train_colbert.py script
accelerate launch \
    --multi_gpu \
    --num_machines=1 \
    --mixed_precision=no \
    --num_processes=8 \
    --main_process_port=$MASTER_PORT \
    hypencoder_cb/train/train_colbert.py \
    $JOB_DIR/config.yaml

# Cleanup RAM after job finishes
rm -rf $JOB_DIR
echo "Cleanup complete."
