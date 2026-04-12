import json
import numpy as np
from scipy import stats
import os

def calculate_ttest(file_ctrl, file_hype, file_teacher, metric='ndcg_cut_10'):
    with open(file_ctrl, 'r') as f:
        data_ctrl = json.load(f)
    with open(file_hype, 'r') as f:
        data_hype = json.load(f)
    with open(file_teacher, 'r') as f:
        data_teacher = json.load(f)
        
    shared_qids = set(data_ctrl.keys()).intersection(set(data_hype.keys())).intersection(set(data_teacher.keys()))
    
    scores_ctrl = []
    scores_hype = []
    scores_teacher = []
    
    for qid in shared_qids:
        if metric in data_ctrl[qid] and metric in data_hype[qid] and metric in data_teacher[qid]:
            scores_ctrl.append(data_ctrl[qid][metric])
            scores_hype.append(data_hype[qid][metric])
            scores_teacher.append(data_teacher[qid][metric])
            
    if not scores_ctrl:
        return None
        
    mean_ctrl = np.mean(scores_ctrl)
    mean_hype = np.mean(scores_hype)
    mean_teacher = np.mean(scores_teacher)
    
    stat, pval = stats.ttest_rel(scores_ctrl, scores_hype)
    
    return {
        "mean_ctrl": mean_ctrl,
        "mean_hype": mean_hype,
        "mean_teacher": mean_teacher,
        "pval": pval,
        "count": len(scores_ctrl)
    }

if __name__ == "__main__":
    datasets = ["covid_results", "dl19_results", "dl20_results", "fiqa_results", "nfcorpus_results", "touche_results"]
    path_hype = "./outputs/inference/hypencoder_teacher_mxbai"
    path_ctrl = "./outputs/inference/control_be_teacher_mxbai"
    path_teacher = "./outputs/inference/mxbai_teacher_baseline"
    
    results = {}
    for ds in datasets:
        f_ctrl = os.path.join(path_ctrl, ds, "metrics/per_query_metrics.json")
        f_hype = os.path.join(path_hype, ds, "metrics/per_query_metrics.json")
        f_teacher = os.path.join(path_teacher, ds, "metrics/per_query_metrics.json")
        
        if all(os.path.exists(f) for f in [f_ctrl, f_hype, f_teacher]):
            res = calculate_ttest(f_ctrl, f_hype, f_teacher)
            if res:
                results[ds] = res
                print(f"Dataset: {ds}")
                print(f"  Teacher: {res['mean_teacher']:.4f}")
                print(f"  Control: {res['mean_ctrl']:.4f}")
                print(f"  Hype:    {res['mean_hype']:.4f}")
                print(f"  p-val:   {res['pval']:.4e}")
                print("-" * 20)
        else:
            print(f"Missing files for {ds}")
