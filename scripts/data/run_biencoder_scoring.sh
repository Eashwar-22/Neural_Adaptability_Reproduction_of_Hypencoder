#!/bin/bash
#SBATCH --job-name="score_bge"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --exclude=mlcbm005,mlcbm007,mlcbm004
#SBATCH --output=logs/score_bge_%A_%a.log

# ... (rest of headers)

source ~/.bashrc
conda activate hype_env

PART_ID=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
INPUT_FILE="data/triples.train.part_${PART_ID}.jsonl"
OUTPUT_FILE="data/triples.train.bge.part_${PART_ID}.jsonl"

echo "Scoring Part $PART_ID: $INPUT_FILE with BAAI/bge-base-en-v1.5..."
echo "Output will be saved to $OUTPUT_FILE"

python -u scripts/data/score_triples_with_biencoder.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --model_name "BAAI/bge-base-en-v1.5" \
    --batch_size 4

echo "Done."
