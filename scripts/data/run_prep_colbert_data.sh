#!/bin/bash
#SBATCH --job-name=prep_colbert_data
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --output=logs/prep_colbert_data_%j.out
#SBATCH --error=logs/prep_colbert_data_%j.err
#SBATCH --exclude=mlcbm015

source .bashrc
conda activate hype_env

cd .

echo "Generating ColBERTv2 teacher scores for 1k triples..."
python scripts/data/prepare_colbert_data.py
echo "Done!"
