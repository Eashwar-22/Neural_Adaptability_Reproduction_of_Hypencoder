#!/bin/bash
#SBATCH --job-name="tokenize"
#SBATCH --partition=cpu-ferranti
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/tokenize_%j.log


echo "Starting Tokenization..."

python hypencoder_cb/utils/tokenizer_utils.py \
  --standard_format_jsonl=data/triples.train.jsonl \
  --output_file=data/triples.train.tokenized.jsonl \
  --tokenizer="google-bert/bert-base-uncased" \
  --add_special_tokens=True \
  --query_max_length=32 \
  --item_max_length=196 \
  --batch_size=1000

echo "Done!"