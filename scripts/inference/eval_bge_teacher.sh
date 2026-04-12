#!/bin/bash
#SBATCH --job-name="eval_bge_tch"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_bge_teacher_%j.log

export CUDA_VISIBLE_DEVICES=0
source ~/.bashrc
conda activate hype_env

MODEL_NAME="BAAI/bge-reranker-v2-m3"
BASE_OUT_DIR="./outputs/inference/bge_v2_m3_teacher_baseline"
BGE_BASE_DIR="./outputs/inference/bge_baseline"

mkdir -p "$BASE_OUT_DIR"

DATASETS=(
    "msmarco-passage/trec-dl-2019/judged:dl19"
    "msmarco-passage/trec-dl-2020/judged:dl20"
    "beir/trec-covid:trec_covid"
    "beir/nfcorpus/test:nfcorpus"
    "beir/fiqa/test:fiqa"
    "beir/webis-touche2020/v2:touche2020"
)

for entry in "${DATASETS[@]}"
do
    IRS_NAME="${entry%%:*}"
    DIR_NAME="${entry##*:}"
    
    echo "------------------------------------------------"
    echo "Evaluating $MODEL_NAME on $IRS_NAME ($DIR_NAME)..."
    python3 scripts/utils/evaluate_crossencoder_rerank.py \
        --model_name "$MODEL_NAME" \
        --candidate_run "$BGE_BASE_DIR/$DIR_NAME/run.json" \
        --ir_dataset_name "$IRS_NAME" \
        --output_dir "$BASE_OUT_DIR/$DIR_NAME" \
        --batch_size 64
done

echo "------------------------------------------------"
echo "All BGE Teacher evaluations complete."
