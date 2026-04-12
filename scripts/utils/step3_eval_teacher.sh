#!/bin/bash
# scripts/utils/step3_eval_teacher.sh

# Set environment
export CUDA_VISIBLE_DEVICES=0
source ~/.bashrc
conda activate hype_env
MODEL_NAME="cross-encoder/ms-marco-MiniLM-L-12-v2"
BASE_OUT_DIR="./outputs/inference/teacher_baseline"
BGE_BASE_DIR="./outputs/inference/bge_baseline"

# 1. TREC DL '19
# echo "------------------------------------------------"
# echo "Evaluating Teacher on TREC DL '19..."
# python3 scripts/utils/evaluate_crossencoder_rerank.py \
#     --model_name "$MODEL_NAME" \
#     --candidate_run "$BGE_BASE_DIR/dl19/run.json" \
#     --ir_dataset_name "msmarco-passage/trec-dl-2019/judged" \
#     --output_dir "$BASE_OUT_DIR/dl19" \
#     --batch_size 64

# 2. TREC DL '20
# echo "------------------------------------------------"
# echo "Evaluating Teacher on TREC DL '20..."
# python3 scripts/utils/evaluate_crossencoder_rerank.py \
#     --model_name "$MODEL_NAME" \
#     --candidate_run "$BGE_BASE_DIR/dl20/run.json" \
#     --ir_dataset_name "msmarco-passage/trec-dl-2020/judged" \
#     --output_dir "$BASE_OUT_DIR/dl20" \
#     --batch_size 64

# 3. TREC-COVID
# echo "------------------------------------------------"
# echo "Evaluating Teacher on TREC-COVID..."
# python3 scripts/utils/evaluate_crossencoder_rerank.py \
#     --model_name "$MODEL_NAME" \
#     --candidate_run "$BGE_BASE_DIR/trec_covid/run.json" \
#     --ir_dataset_name "beir/trec-covid" \
#     --output_dir "$BASE_OUT_DIR/trec_covid" \
#     --batch_size 64

# 4. NFCorpus
echo "------------------------------------------------"
echo "Evaluating Teacher on NFCorpus..."
python3 scripts/utils/evaluate_crossencoder_rerank.py \
    --model_name "$MODEL_NAME" \
    --candidate_run "$BGE_BASE_DIR/nfcorpus/run.json" \
    --ir_dataset_name "beir/nfcorpus/test" \
    --output_dir "$BASE_OUT_DIR/nfcorpus" \
    --batch_size 64

# 5. FiQA
echo "------------------------------------------------"
echo "Evaluating Teacher on FiQA..."
python3 scripts/utils/evaluate_crossencoder_rerank.py \
    --model_name "$MODEL_NAME" \
    --candidate_run "$BGE_BASE_DIR/fiqa/run.json" \
    --ir_dataset_name "beir/fiqa/test" \
    --output_dir "$BASE_OUT_DIR/fiqa" \
    --batch_size 64

# 6. Touché v2
echo "------------------------------------------------"
echo "Evaluating Teacher on Touché v2..."
python3 scripts/utils/evaluate_crossencoder_rerank.py \
    --model_name "$MODEL_NAME" \
    --candidate_run "$BGE_BASE_DIR/touche2020/run.json" \
    --ir_dataset_name "beir/webis-touche2020/v2" \
    --output_dir "$BASE_OUT_DIR/touche" \
    --batch_size 64

# 7. DBPedia
echo "------------------------------------------------"
echo "Evaluating Teacher on DBPedia..."
# Note: DBPedia BGE run must be completed first
python3 scripts/utils/evaluate_crossencoder_rerank.py \
    --model_name "$MODEL_NAME" \
    --candidate_run "$BGE_BASE_DIR/dbpedia/run.json" \
    --ir_dataset_name "beir/dbpedia-entity/test" \
    --output_dir "$BASE_OUT_DIR/dbpedia" \
    --batch_size 64

echo "------------------------------------------------"
echo "All Teacher evaluations complete."
