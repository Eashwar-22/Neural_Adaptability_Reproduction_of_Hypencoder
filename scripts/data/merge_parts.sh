#!/bin/bash
#SBATCH --job-name="merge_data"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/merge_%j.log

source ~/.bashrc

TYPE=$1

if [ "$TYPE" == "biencoder" ]; then
    echo "Merging Bi-Encoder parts..."
    cat data/triples.train.bge.part_*.jsonl > data/triples.train.bge.jsonl
    echo "Created data/triples.train.bge.jsonl"
elif [ "$TYPE" == "crossencoder" ]; then
    echo "Merging Cross-Encoder parts..."
    cat data/triples.train.crossencoder.part_*.jsonl > data/triples.train.crossencoder.jsonl
    echo "Created data/triples.train.crossencoder.jsonl"
else
    echo "Unknown type: $TYPE"
    exit 1
fi

echo "Done."
