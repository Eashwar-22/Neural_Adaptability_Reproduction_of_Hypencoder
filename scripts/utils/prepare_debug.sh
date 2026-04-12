#!/bin/bash
#SBATCH --job-name="prep_debug"
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/prep_debug_%j.log

source ~/.bashrc
# Direct path to your environment's python
PYTHON_EXE="/mnt/lustre/work/eickhoff/esx510/hype_env/bin/python"

echo "1. Creating Debug Dataset (First 5000 lines)..."
head -n 5000 data/triples.train.jsonl > data/debug.train.jsonl

echo "2. Tokenizing Debug Dataset..."
$PYTHON_EXE hypencoder_cb/utils/tokenizer_utils.py \
  --standard_format_jsonl=data/debug.train.jsonl \
  --output_file=data/debug.train.tokenized.jsonl \
  --tokenizer="google-bert/bert-base-uncased" \
  --add_special_tokens=True \
  --query_max_length=32 \
  --item_max_length=196 \
  --batch_size=1000

echo "Done!"