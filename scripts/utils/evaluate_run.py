
import json
import ir_datasets
import pytrec_eval
import sys
import os

def evaluate(run_file, dataset_name):
    print(f"Loading run from {run_file}...")
    run = {}
    with open(run_file, 'r') as f:
        for line in f:
            obj = json.loads(line)
            # The format is {"query": {"id": ...}, "items": [{"id": ..., "score": ...}, ...]}
            
            # Access query ID safely
            if 'query' in obj and 'id' in obj['query']:
                qid = str(obj['query']['id'])
            else:
                # Fallback or skip if malformed
                continue
                
            if qid not in run:
                run[qid] = {}
            
            # Iterate over retrieved items
            if 'items' in obj:
                for item in obj['items']:
                    doc_id = str(item['id'])
                    score = float(item['score'])
                    run[qid][doc_id] = score
            
    print(f"Loading qrels from {dataset_name}...")
    dataset = ir_datasets.load(dataset_name)
    qrels = {}
    for qrel in dataset.qrels_iter():
        qid = str(qrel.query_id)
        doc_id = str(qrel.doc_id)
        rel = int(qrel.relevance)
        
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][doc_id] = rel
        
    print("Evaluating...")
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut.10', 'recall.1000', 'map'})
    results = evaluator.evaluate(run)
    
    # Aggregate
    aggregated = {
        'ndcg_cut_10': 0.0,
        'recall_1000': 0.0,
        'map': 0.0,
        'count': 0
    }
    
    for qid, measures in results.items():
        aggregated['count'] += 1
        aggregated['ndcg_cut_10'] += measures['ndcg_cut_10']
        aggregated['recall_1000'] += measures['recall_1000']
        aggregated['map'] += measures['map']
        
    cnt = aggregated['count']
    if cnt > 0:
        print(f"nDCG@10: {aggregated['ndcg_cut_10'] / cnt:.4f}")
        print(f"R@1000:  {aggregated['recall_1000'] / cnt:.4f}")
        print(f"MRR (MAP approx):   {aggregated['map'] / cnt:.4f}")
    else:
        print("No intersecting queries found between run and qrels.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate_run.py <run_jsonl> <ir_dataset_name>")
        # Default for this session
        run_path = "outputs/inference/hypencoder_retrained_improved/nfcorpus_results/retrieved_items.jsonl"
        ir_name = "beir/nfcorpus/test"
        if os.path.exists(run_path):
             evaluate(run_path, ir_name)
        else:
             sys.exit(1)
    else:
        evaluate(sys.argv[1], sys.argv[2])
