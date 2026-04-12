
import os
import json
import argparse
import collections
from typing import Dict, List, Tuple
import torch
import ir_datasets
from tqdm import tqdm
from hypencoder_cb.utils.eval_utils import do_eval_and_pretty_print

def load_run(path: str) -> Dict[str, Dict[str, float]]:
    """
    Loads a run from a JSONL file (retrieved_items.jsonl).
    Format: {"query_id": "...", "doc_id": "...", "score": ...}
    Returns: {query_id: {doc_id: score}}
    """
    run = collections.defaultdict(dict)
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            run[item['query_id']][item['doc_id']] = item['score']
    return run

def normalize_scores(run: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Min-Max normalizes scores for each query to [0, 1].
    This is crucial when combining models with different score distributions.
    """
    normalized_run = collections.defaultdict(dict)
    for qid, docs in run.items():
        if not docs:
            continue
        scores = list(docs.values())
        min_s = min(scores)
        max_s = max(scores)
        if max_s == min_s:
            for doc_id in docs:
                normalized_run[qid][doc_id] = 1.0
        else:
            for doc_id, score in docs.items():
                normalized_run[qid][doc_id] = (score - min_s) / (max_s - min_s)
    return normalized_run

def fuse_runs(
    run1: Dict[str, Dict[str, float]], 
    run2: Dict[str, Dict[str, float]], 
    alpha: float = 0.5,
    normalize: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Fuses two runs: score = alpha * run1 + (1 - alpha) * run2
    If a doc is missing in one run, it gets the minimum score of that run (or 0 if normalized).
    """
    if normalize:
        print("Normalizing scores...")
        run1 = normalize_scores(run1)
        run2 = normalize_scores(run2)

    fused_run = collections.defaultdict(dict)
    all_qids = set(run1.keys()) | set(run2.keys())

    for qid in all_qids:
        docs1 = run1.get(qid, {})
        docs2 = run2.get(qid, {})
        all_docs = set(docs1.keys()) | set(docs2.keys())
        
        # Get defaults for missing docs (min score or 0)
        min1 = min(docs1.values()) if docs1 else 0.0
        min2 = min(docs2.values()) if docs2 else 0.0

        for doc_id in all_docs:
            s1 = docs1.get(doc_id, min1)
            s2 = docs2.get(doc_id, min2)
            
            fused_score = (alpha * s1) + ((1.0 - alpha) * s2)
            fused_run[qid][doc_id] = fused_score
            
    return fused_run

def save_run(run: Dict[str, Dict[str, float]], output_path: str):
    with open(output_path, 'w') as f:
        for qid, docs in run.items():
            # Sort by score descending
            sorted_docs = sorted(docs.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, score) in enumerate(sorted_docs):
                item = {
                    "query_id": qid,
                    "doc_id": doc_id,
                    "rank": rank + 1,
                    "score": score
                }
                f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run1", type=str, required=True, help="Path to first run (e.g. Hypen-ColBERT)")
    parser.add_argument("--run2", type=str, required=True, help="Path to second run (e.g. Hypencoder)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for run1")
    parser.add_argument("--dataset", type=str, required=True, help="IR Dataset name for eval")
    parser.add_argument("--normalize", action="store_true", help="Apply MinMax normalization before fusion")
    args = parser.parse_args()

    print(f"Loading Run 1: {args.run1}")
    r1 = load_run(args.run1)
    print(f"Loading Run 2: {args.run2}")
    r2 = load_run(args.run2)

    print(f"Fusing with alpha={args.alpha}...")
    fused = fuse_runs(r1, r2, alpha=args.alpha, normalize=args.normalize)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "retrieved_items.jsonl")
    save_run(fused, output_file)
    print(f"Saved fused run to {output_file}")

    print("Running Evaluation...")
    do_eval_and_pretty_print(
        ir_dataset_name=args.dataset,
        retrieval_path=output_file,
        output_dir=os.path.join(args.output_dir, "metrics"),
        metric_names=["nDCG@10", "nDCG@1000", "R@1000", "MRR"]
    )
