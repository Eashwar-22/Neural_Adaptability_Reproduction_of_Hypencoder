#!/bin/bash
#SBATCH --job-name=prep_bm25
#SBATCH --output=logs/prep_bm25_%j.log
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00

source ~/.bashrc
conda activate hype_env
export PYTHONPATH=.
export HF_HOME="${HYPENCODER_CACHE:-./cache}"

# Ensure rank_bm25 is installed
cd .
pip install rank_bm25

echo "Starting BM25 data preparation..."
echo "This will build a BM25 index over ~8.8M MS MARCO passages"
echo "and score training triples. Expect ~2-6 hours."

python scripts/utils/prepare_bm25_data.py \
    --output_path data/triples.train.bm25.jsonl \
    --bert_tokenizer google-bert/bert-base-uncased

echo "Data preparation complete."
echo "Validating output..."
wc -l data/triples.train.bm25.jsonl
head -n 1 data/triples.train.bm25.jsonl | python3 -m json.tool | head -n 20
