#!/bin/bash
#SBATCH --job-name="bge_base"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/inference_bge_baseline_%j.log

source ~/.bashrc
conda activate hype_env
export PYTHONPATH=.

# BGE Base from Hugging Face
MODEL_NAME="BAAI/bge-base-en-v1.5"
BASE_OUT_DIR="outputs/inference/bge_baseline"

# Array of datasets to test
# Format: "ir_dataset_key output_folder_name"
DATASETS=(
    "beir/nfcorpus nfcorpus"
    "beir/trec-covid trec_covid"
    "beir/webis-touche2020 touche2020"
    "beir/fiqa fiqa"
    "msmarco-passage/trec-dl-2019/judged dl19"
    "msmarco-passage/trec-dl-2020/judged dl20"
    "beir/dbpedia-entity/test dbpedia"
)

echo "Starting BGE Baseline Evaluation..."
echo "Model: $MODEL_NAME"

for entry in "${DATASETS[@]}"; do
    set -- $entry
    IR_KEY=$1
    OUT_NAME=$2
    
    OUT_DIR="${BASE_OUT_DIR}/${OUT_NAME}"
    
    echo "------------------------------------------------"
    echo "Processing $IR_KEY -> $OUT_DIR"
    
    python scripts/inference/run_bge_eval.py \
        --model_name "$MODEL_NAME" \
        --ir_dataset_name "$IR_KEY" \
        --output_dir "$OUT_DIR" \
        --batch_size 2048
        
done

echo "All BGE evaluations completed."
