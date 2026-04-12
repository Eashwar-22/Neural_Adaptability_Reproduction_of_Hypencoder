#!/bin/bash
#SBATCH --job-name=mxbai_score_prep
#SBATCH --output=logs/run_prep_mxbai_reranker_%j.out
#SBATCH --error=logs/run_prep_mxbai_reranker_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1

# Load Conda environment 
source ~/.bashrc
conda activate hype_env

# Set PYTHONPATH so absolute imports work
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Starting mxbai-rerank-large-v1 Triplet Scoring at $(date)"

# Run the python script directly pointing from the root workspace
python scripts/utils/prepare_bge_reranker_data.py \
    --input_path "data/triples.train.jsonl" \
    --output_path "data/triples.train.mxbai.jsonl" \
    --model_name "mixedbread-ai/mxbai-rerank-large-v1" \
    --batch_size 512

echo "Finished mxbai Scoring at $(date)"
