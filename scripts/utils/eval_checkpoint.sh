#!/bin/bash
#SBATCH --job-name="hype_eval"
#SBATCH --partition=a100-galvani
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_%j.log

source ~/.bashrc
# Activate your environment
source /mnt/lustre/work/eickhoff/esx510/hype_env/bin/activate

# Path to where the trainer is saving the model
CHECKPOINT_DIR="checkpoints/hypencoder.6_layer_frozen"

echo "Starting Evaluation on Checkpoint: $CHECKPOINT_DIR"

/mnt/lustre/work/eickhoff/esx510/hype_env/bin/python hypencoder_cb/inference/approx_retrieve.py \
    --query_file=data/queries.dev.jsonl \
    --output_file=logs/results_dev_frozen.json \
    --checkpoint_path=$CHECKPOINT_DIR \
    --batch_size=1000 \
    --top_k=100 

echo "Evaluation Complete."