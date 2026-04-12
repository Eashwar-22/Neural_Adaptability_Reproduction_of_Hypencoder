"""
BGE-Reranker-v2-m3 Data Preparation Script for Hypencoder Distillation

Generates State-of-the-Art (SOTA) cross-encoder scored MS MARCO training triples 
in the JSONL format required by the Hypencoder Margin-MSE training pipeline.

Usage:
    python scripts/utils/prepare_bge_reranker_data.py \
        --input_path data/triples.train.jsonl \
        --output_path data/triples.train.bge_v2_m3.jsonl \
        --model_name BAAI/bge-reranker-v2-m3 \
        --batch_size 256
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def score_pairs_batch(model, tokenizer, pairs: list[tuple[str, str]], batch_size: int, device: str) -> list[float]:
    """Score a batch of (query, passage) pairs using the Cross-Encoder."""
    all_scores = []
    
    # Process in sub-batches to avoid OOM
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i + batch_size]
        
        # BGE-Reranker tokenizer handles the [CLS] Query [SEP] Passage [SEP] format automatically
        inputs = tokenizer(batch_pairs, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # DataParallel often returns tuples rather than dictionaries, so handle both
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            
            # The model outputs a single logit per pair representing relevance
            scores = logits.view(-1,).float().cpu().tolist()
            
        all_scores.extend(scores)
        
    return all_scores


def main():
    parser = argparse.ArgumentParser(description="Generate SOTA Cross-Encoder scored training triples")
    parser.add_argument("--input_path", type=str, default="data/triples.train.jsonl",
                        help="Path to the original training triples JSONL file")
    parser.add_argument("--output_path", type=str, default="data/triples.train.bge_v2_m3.jsonl",
                        help="Path to save the new scored triples")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-reranker-v2-m3",
                        help="HuggingFace model ID for the Cross-Encoder teacher")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for inference. Adjust based on GPU memory.")
    parser.add_argument("--max_triples", type=int, default=None,
                        help="Maximum number of triples to process (for debugging)")
    args = parser.parse_args()

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load Teacher Model
    logger.info(f"Loading Cross-Encoder Teacher: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Using float16 for memory efficiency on modern GPUs
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, torch_dtype=torch.float16)
    model.to(device)
    
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = torch.nn.DataParallel(model)
        
    model.eval()

    # File mapping
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading from: {input_path}")
    logger.info(f"Writing to: {output_path}")

    # Count total lines for tqdm
    logger.info("Counting lines in input file...")
    with open(input_path, "r") as f:
        total_lines = sum(1 for _ in f)
    if args.max_triples:
        total_lines = min(total_lines, args.max_triples)

    written = 0
    skipped = 0

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for i, line in enumerate(tqdm(fin, total=total_lines, desc="Scoring MS MARCO Triples")):
            if args.max_triples and i >= args.max_triples:
                break
            
            try:
                record = json.loads(line)
                
                query_text = record["query"]["content"]
                
                if not record.get("items"):
                    skipped += 1
                    continue

                # 1. Gather all pairs for this query to score them efficiently in a batch
                pairs_to_score = []
                for item in record["items"]:
                    pairs_to_score.append((query_text, item["content"]))
                    
                # 2. Score all pairs for this query
                scores = score_pairs_batch(model, tokenizer, pairs_to_score, args.batch_size, device)
                
                # 3. Re-assign scores back to the JSON object
                for item, score in zip(record["items"], scores):
                    item["score"] = score

                # Write updated record
                fout.write(json.dumps(record) + "\n")
                written += 1

            except json.JSONDecodeError:
                skipped += 1
                continue
            except Exception as e:
                logger.error(f"Error processing line {i}: {e}")
                skipped += 1
                continue

    logger.info(f"Done! Written {written:,} records, skipped {skipped:,}")
    logger.info(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()
