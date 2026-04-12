#!/bin/bash
#SBATCH --job-name="hype_debug"
#SBATCH --partition=a100-galvani
#SBATCH --nodes=1
#SBATCH --gres=gpu:1             # 1 GPU is enough for debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00          # 30 minute limit
#SBATCH --output=logs/debug_log_%j.log

# REMOVED the failing activate line
# source /mnt/lustre/work/eickhoff/esx510/hype_env/bin/activate

echo "Starting DEBUG Training..."

# ADDED Direct path to python executable
/mnt/lustre/work/eickhoff/esx510/hype_env/bin/python hypencoder_cb/train/train.py \
    my_configs/debug_run.yaml