import argparse
import os
import json
import logging
import pytrec_eval
import numpy as np
from scipy import stats
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def load_qrels(qrel_file):
    # Depending on format: IR_datasets or standard trec qrels
    # We will try to parse standard trec qrels if provided, otherwise assume IR_datasets handles it
    qrels = defaultdict(dict)
    if not os.path.exists(qrel_file):
        logging.warning(f"Qrels file not found: {qrel_file}")
        return qrels
    with open(qrel_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                query_id, _, doc_id, rel = parts
                qrels[query_id][doc_id] = int(rel)
    return qrels

def load_run(run_file):
    run = defaultdict(dict)
    if run_file.endswith('.json'):
        with open(run_file, 'r') as f:
            run = json.load(f)
    elif run_file.endswith('.trec'):
        with open(run_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    query_id, _, doc_id, rank, score, _ = parts
                    run[query_id][doc_id] = float(score)
    else:
        raise ValueError("Run file must be .json or .trec")
    return run

def evaluate_run(run, qrels, metrics={'ndcg_cut_10'}):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    return evaluator.evaluate(run)

def get_shared_queries(res1, res2):
    return set(res1.keys()).intersection(set(res2.keys()))

def run_significance_test(scores1, scores2, method='ttest'):
    if len(scores1) != len(scores2) or len(scores1) == 0:
         return None, None
    
    if method == 'ttest':
        stat, pval = stats.ttest_rel(scores1, scores2)
    elif method == 'wilcoxon':
        stat, pval = stats.wilcoxon(scores1, scores2)
    else:
        raise ValueError("method must be 'ttest' or 'wilcoxon'")
        
    return stat, pval

def main():
    parser = argparse.ArgumentParser(description="Run statistical significance tests between two retrieval runs.")
    parser.add_argument("--run1", type=str, required=True, help="Path to first run file (.trec or .json). Typically the baseline/control.")
    parser.add_argument("--run2", type=str, required=True, help="Path to second run file (.trec or .json). Typically the experimental model.")
    parser.add_argument("--qrels", type=str, required=True, help="Path to qrels file.")
    parser.add_argument("--metric", type=str, default="ndcg_cut_10", help="Metric to compare (e.g., ndcg_cut_10, recip_rank)")
    parser.add_argument("--test", type=str, default="ttest", choices=['ttest', 'wilcoxon'], help="Test to run")
    
    args = parser.parse_args()
    
    logging.info(f"Loading qrels from {args.qrels}")
    qrels = load_qrels(args.qrels)
    if not qrels:
        logging.error("Failed to load qrels. Exiting.")
        return
        
    logging.info(f"Loading run1 from {args.run1}")
    run1 = load_run(args.run1)
    
    logging.info(f"Loading run2 from {args.run2}")
    run2 = load_run(args.run2)
    
    logging.info(f"Evaluating runs using {args.metric}")
    eval1 = evaluate_run(run1, qrels, {args.metric})
    eval2 = evaluate_run(run2, qrels, {args.metric})
    
    shared_queries = get_shared_queries(eval1, eval2)
    logging.info(f"Found {len(shared_queries)} shared queries between runs.")
    
    scores1 = []
    scores2 = []
    
    for qid in shared_queries:
        if args.metric in eval1[qid] and args.metric in eval2[qid]:
            scores1.append(eval1[qid][args.metric])
            scores2.append(eval2[qid][args.metric])
            
    if not scores1:
        logging.error("No valid scores found for the specified metric.")
        return
        
    mean1 = np.mean(scores1)
    mean2 = np.mean(scores2)
    delta = mean2 - mean1
    
    logging.info(f"Run1 Mean {args.metric}: {mean1:.4f}")
    logging.info(f"Run2 Mean {args.metric}: {mean2:.4f}")
    logging.info(f"Delta: {delta:.4f}")
    
    stat, pval = run_significance_test(scores1, scores2, args.test)
    logging.info(f"Test ({args.test}): p-value = {pval:.4e}")
    
    if pval < 0.05:
        logging.info("Result is STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        logging.info("Result is NOT statistically significant (p >= 0.05)")

if __name__ == "__main__":
    main()
