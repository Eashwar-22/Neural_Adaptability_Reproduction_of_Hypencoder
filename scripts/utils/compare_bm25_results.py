import json
import argparse
import numpy as np
from scipy import stats
import os

def calculate_ttest(file1, file2, metric='ndcg_cut_10'):
    with open(file1, 'r') as f:
        data1 = json.load(f)
    with open(file2, 'r') as f:
        data2 = json.load(f)
        
    shared_qids = set(data1.keys()).intersection(set(data2.keys()))
    
    scores1 = []
    scores2 = []
    
    for qid in shared_qids:
        if metric in data1[qid] and metric in data2[qid]:
            scores1.append(data1[qid][metric])
            scores2.append(data2[qid][metric])
            
    if not scores1:
        print(f"No matching queries found for metric {metric}")
        return
        
    mean1 = np.mean(scores1)
    mean2 = np.mean(scores2)
    stat, pval = stats.ttest_rel(scores1, scores2)
    
    basename = os.path.basename(os.path.dirname(os.path.dirname(file1)))
    
    print(f"Dataset: {basename}")
    print(f"Metric: {metric}")
    print(f"Control Mean: {mean1:.4f}")
    print(f"Hype Mean:    {mean2:.4f}")
    print(f"Delta:        {mean2 - mean1:.4f}")
    print(f"Queries:      {len(scores1)}")
    print(f"p-value:      {pval:.4e}")
    print("-" * 30)

if __name__ == "__main__":
    import sys
    # Hardcoded list for our current run
    datasets = ["covid_results", "dl19_results", "dl20_results", "fiqa_results", "nfcorpus_results", "touche_results"]
    base_path_hype = "./outputs/inference/distill_bm25"
    base_path_ctrl = "./outputs/inference/control_be_bm25"
    
    for ds in datasets:
        f1 = os.path.join(base_path_ctrl, ds, "metrics/per_query_metrics.json")
        f2 = os.path.join(base_path_hype, ds, "metrics/per_query_metrics.json")
        if os.path.exists(f1) and os.path.exists(f2):
            calculate_ttest(f1, f2)
        else:
            print(f"Missing files for {ds}")
