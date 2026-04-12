"""
BM25 Data Preparation Script for Hypencoder Distillation

Generates BM25-scored MS MARCO training triples in the same JSONL format
used by the existing Hypencoder training pipeline.

Usage:
    python scripts/utils/prepare_bm25_data.py \
        --output_path data/triples.train.bm25.jsonl \
        --max_triples 500000
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import ir_datasets
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def tokenize_for_bm25(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenization for BM25."""
    return text.lower().split()


def build_bm25_index(corpus_iter, doc_count: int):
    """Build a BM25 index from an ir_datasets corpus."""
    logger.info(f"Building BM25 index over {doc_count:,} documents...")

    doc_ids = []
    tokenized_corpus = []

    for doc in tqdm(corpus_iter, total=doc_count, desc="Indexing"):
        doc_ids.append(doc.doc_id)
        tokenized_corpus.append(tokenize_for_bm25(doc.text))

    logger.info("Fitting BM25 model...")
    bm25 = BM25Okapi(tokenized_corpus)

    # Build doc_id -> index mapping
    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    logger.info(f"BM25 index built. {len(doc_ids):,} documents indexed.")
    return bm25, doc_id_to_idx, doc_ids


def score_document(bm25: BM25Okapi, query_tokens: list[str], doc_idx: int) -> float:
    """Score a single document against a query using BM25, optimized for speed."""
    score = 0.0
    doc_len = bm25.doc_len[doc_idx]
    doc_freqs = bm25.doc_freqs[doc_idx]
    
    for q in query_tokens:
        if q not in doc_freqs:
            continue
        q_freq = doc_freqs[q]
        idf = bm25.idf.get(q) or 0.0
        
        numerator = q_freq * (bm25.k1 + 1)
        denominator = q_freq + bm25.k1 * (1 - bm25.b + bm25.b * doc_len / bm25.avgdl)
        score += idf * (numerator / denominator)
        
    return score


def main():
    parser = argparse.ArgumentParser(description="Generate BM25-scored training triples")
    parser.add_argument("--triples_path", type=str, default="data/triples.train.jsonl",
                        help="Path to local training triples JSONL file")
    parser.add_argument("--output_path", type=str, default="data/triples.train.bm25.jsonl",
                        help="Path to write BM25-scored output JSONL")
    parser.add_argument("--bert_tokenizer", type=str, default="google-bert/bert-base-uncased",
                        help="BERT tokenizer (unused, kept for compatibility)")
    parser.add_argument("--max_triples", type=int, default=None,
                        help="Max number of triples to process (None = all)")
    args = parser.parse_args()

    # Load MS MARCO corpus for BM25 indexing (this part was fast)
    logger.info("Loading MS MARCO passage corpus for BM25 index...")
    corpus_dataset = ir_datasets.load("msmarco-passage")
    doc_count = corpus_dataset.docs_count()

    # Build BM25 index
    bm25, doc_id_to_idx, doc_ids = build_bm25_index(corpus_dataset.docs_iter(), doc_count)

    # Note: BERT tokenizer arg is accepted for compatibility but not used
    # BM25 uses simple whitespace tokenization

    # Process local triples file
    input_path = Path(args.triples_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing triples from {input_path}...")
    written = 0
    skipped = 0

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for i, line in enumerate(tqdm(fin, desc="Scoring triples")):
            if args.max_triples and i >= args.max_triples:
                break
            
            try:
                record = json.loads(line)
                
                # Extract query
                query_text = record["query"]["content"]
                # query_id = record["query"]["id"] # Not strictly needed for scoring but good to have
                query_tokens = tokenize_for_bm25(query_text)
                
                # Check if items exist
                if not record.get("items") or len(record["items"]) < 2:
                    skipped += 1
                    continue

                # Score items
                valid_items = []
                for item in record["items"]:
                    doc_id = item["id"]
                    if doc_id in doc_id_to_idx:
                        bm25_score = score_document(bm25, query_tokens, doc_id_to_idx[doc_id])
                        item["score"] = bm25_score 
                        valid_items.append(item)
                        
                # Update record
                record["items"] = valid_items 
                
                # Filter if not enough valid items left
                if len(record["items"]) < 2:
                    skipped += 1
                    continue

                fout.write(json.dumps(record) + "\n")
                written += 1

            except json.JSONDecodeError:
                skipped += 1
                continue

    logger.info(f"Done! Written {written:,} records, skipped {skipped:,}")
    logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
