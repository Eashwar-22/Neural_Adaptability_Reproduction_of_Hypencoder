#!/bin/bash
#SBATCH --job-name="hype_lora_r64_a256"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_8gpu_lora_r64_alpha256_%j.log

# Move to the project root directory
cd ./

echo "Current directory: $(pwd)"

# Environment variables
export HF_HOME="${HYPENCODER_CACHE:-./cache}"
export HF_DATASETS_CACHE="${HYPENCODER_CACHE:-./cache}"
export TRANSFORMERS_CACHE="${HYPENCODER_CACHE:-./cache}"

# Create RAM disk for dataset to speed up loading
# Using job ID to ensuring unique path
RAM_DISK_PATH="/dev/shm/hype_full_$SLURM_JOBID"
mkdir -p $RAM_DISK_PATH

# Cleanup function to always remove RAM disk
cleanup() {
    echo "Cleaning up RAM disk at $RAM_DISK_PATH"
    rm -rf $RAM_DISK_PATH
}
trap cleanup EXIT

echo "Copying dataset to RAM disk..."
cp -r "data/triples.train.tokenized.jsonl" "$RAM_DISK_PATH/"

echo "Starting training..."
# Using the NEW Alpha 256 config file
# Using `torchrun` for DDP (multi-GPU)
# Setting OMP_NUM_THREADS to avoid overhead
export OMP_NUM_THREADS=4

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --rdzv_endpoint=localhost:$((RANDOM + 10000)) \
    hypencoder_cb/train/train.py \
    --config_path="my_configs/hypencoder.6_layer_lora_r64_alpha256.yaml" \
    --data_config.training_data_jsonl="$RAM_DISK_PATH/triples.train.tokenized.jsonl"
