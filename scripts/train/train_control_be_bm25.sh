#!/bin/bash
#SBATCH --job-name=ctrl_bm25
#SBATCH --output=logs/ctrl_bm25_%j.log
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate hype_env
export PYTHONPATH=.
export HF_HOME="${HYPENCODER_CACHE:-./cache}"

cd .

echo "Starting BM25 Control Bi-Encoder Training..."
echo "Config: configs/control_be_bm25.yaml"

torchrun --nproc_per_node=8 \
    hypencoder_cb/train/train.py \
    --config_path configs/control_be_bm25.yaml

echo "Training complete."
