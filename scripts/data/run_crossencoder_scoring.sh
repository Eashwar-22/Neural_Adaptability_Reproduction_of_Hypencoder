#!/bin/bash
#SBATCH --job-name="score_ce"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --exclude=mlcbm005,mlcbm007,mlcbm004
#SBATCH --output=logs/score_ce_%A_%a.log

source ~/.bashrc
conda activate hype_env

PART_ID=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
INPUT_FILE="data/triples.train.part_${PART_ID}.jsonl"
OUTPUT_FILE="data/triples.train.crossencoder.part_${PART_ID}.jsonl"
MODEL_NAME="BAAI/bge-reranker-v2-m3"

echo "Scoring Part $PART_ID: $INPUT_FILE with $MODEL_NAME..."
echo "Output will be saved to $OUTPUT_FILE"

python -u scripts/data/score_triples_with_crossencoder.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --model_name "$MODEL_NAME" \
    --batch_size 32

echo "Done."
