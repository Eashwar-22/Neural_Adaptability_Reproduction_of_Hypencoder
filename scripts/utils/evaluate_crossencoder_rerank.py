
import os
import json
import argparse
import torch
import ir_datasets
import pytrec_eval
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
from collections import defaultdict

def load_cross_encoder(model_name: str, device: str = "cuda"):
    print(f"Loading Cross-Encoder: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)
    # Enable FP16 if available
    if device == "cuda":
        model.half()
    return model, tokenizer

def load_run_file(run_path: str):
    """
    Loads a run file (TREC or JSON).
    Returns a dict: {qid: {docid: score}}
    """
    results = defaultdict(dict)
    run_path = Path(run_path)
    
    if run_path.suffix == '.json':
        with open(run_path, 'r') as f:
            data = json.load(f)
            # JSON format: {qid: {docid: score}}
            return data
    else:
        # TREC format: qid Q0 docid rank score run_name
        with open(run_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    qid, _, docid, _, score, _ = parts[:6]
                    results[qid][docid] = float(score)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    parser.add_argument("--candidate_run", type=str, required=True, help="Path to run.json or run.trec to re-rank")
    parser.add_argument("--ir_dataset_name", type=str, required=True, help="e.g. beir/nfcorpus")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128) # Smaller batch size for CE
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--top_k_candidates", type=int, default=1000, help="Number of candidates to re-rank per query")
    args = parser.parse_args()

    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Candidates
    print(f"Loading candidates from: {args.candidate_run}")
    candidates = load_run_file(args.candidate_run)
    print(f"Loaded {len(candidates)} queries with candidates.")

    # 2. Load Dataset (Text)
    print(f"Loading dataset: {args.ir_dataset_name}")
    dataset = ir_datasets.load(args.ir_dataset_name)
    
    # Create lookups
    docs_store = dataset.docs_store()
    
    # Queries lookup - ir_datasets iterators can be slow, might be better to load all if small
    # But usually queries_store is not standard.
    # We will iterate queries and match with our candidates.
    
    queries = {}
    print("Reading queries...")
    for query in dataset.queries_iter():
        queries[query.query_id] = query.text

    # 3. Load Model
    model, tokenizer = load_cross_encoder(args.model_name, args.device)
    
    # 4. Re-rank
    new_results = {}
    
    # Prepare all pairs for batch processing
    # To save memory, we can process query-by-query or in chunks. Evaluator needs full result dict though.
    # Let's process in large chunks of pairs.
    
    all_pairs_tasks = [] # (qid, docid, query_text, doc_text)
    
    print("Preparing pairs...")
    
    # Filter valid QIDs
    valid_qids = [qid for qid in candidates.keys() if qid in queries]
    print(f"Processing {len(valid_qids)} queries (subset of candidates that exist in dataset).")

    for qid in tqdm(valid_qids):
        query_text = queries[qid]
        
        # Get top-k candidates only
        doc_scores = candidates[qid]
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:args.top_k_candidates]
        
        for docid, _ in sorted_docs:
            try:
                # doc is usually NamedTuple, get valid text field
                doc = docs_store.get(docid)
                if not doc:
                    continue
                # Construct passage text (Title + Text)
                # Handle different schemas
                doc_text = ""
                if hasattr(doc, 'title') and doc.title:
                    doc_text += doc.title + " "
                if hasattr(doc, 'text'):
                    doc_text += doc.text
                elif hasattr(doc, 'body'):
                    doc_text += doc.body
                
                all_pairs_tasks.append((qid, docid, query_text, doc_text))
                
            except KeyError:
                continue

    print(f"Total pairs to score: {len(all_pairs_tasks)}")
    
    # Run Inference
    batch_size = args.batch_size
    
    # We will write results directly to dict
    # Re-initialize new_results with inner dicts
    for qid in valid_qids:
        new_results[qid] = {}

    print("Scoring pairs...")
    with torch.no_grad():
        for i in tqdm(range(0, len(all_pairs_tasks), batch_size)):
            batch = all_pairs_tasks[i : i + batch_size]
            
            # format: [[q, d], [q, d], ...]
            text_pairs = [[b[2], b[3]] for b in batch]
            
            inputs = tokenizer(
                text_pairs, 
                padding=True, 
                truncation=True, 
                max_length=args.max_length, 
                return_tensors="pt"
            ).to(args.device)
            
            logits = model(**inputs).logits
            scores = logits.view(-1).float().cpu().numpy()
            
            # Store scores
            for j, score in enumerate(scores):
                qid, docid, _, _ = batch[j]
                new_results[qid][docid] = float(score)

    # 5. Save Output
    print("Saving re-ranked results...")
    run_file = output_dir / "reranked_run.json"
    with open(run_file, "w") as f:
        json.dump(new_results, f)
        
    trec_file = output_dir / "reranked_run.trec"
    with open(trec_file, "w") as f:
        for qid, docs in new_results.items():
            rank = 1
            sorted_docs = sorted(docs.items(), key=lambda item: item[1], reverse=True)
            for docid, score in sorted_docs:
                f.write(f"{qid} Q0 {docid} {rank} {score:.4f} CrossEncoder\n")
                rank += 1

    # 6. Evaluate
    print("Evaluating...")
    qrels = {}
    try:
        # qrels_iter might be slow, usually loading all qrels is fine for these datasets
        for qrel in dataset.qrels_iter():
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = {}
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
    except Exception as e:
        print(f"Error loading qrels: {e}")
        return

    if not qrels:
        print("No qrels found.")
        return

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut_10', 'map_cut_1000', 'recip_rank', 'recall_1000'})
    metrics = evaluator.evaluate(new_results)
    
    aggregated = {
        'nDCG@10': np.mean([v['ndcg_cut_10'] for v in metrics.values() if 'ndcg_cut_10' in v]),
        'MRR': np.mean([v['recip_rank'] for v in metrics.values() if 'recip_rank' in v]),
        'R@1000': np.mean([v['recall_1000'] for v in metrics.values() if 'recall_1000' in v])
    }
    
    print("================ TEACHER RESULTS ================")
    for k, v in aggregated.items():
        print(f"{k}: {v:.4f}")
    
    with open(output_dir / "aggregated_metrics.txt", "w") as f:
        for k, v in aggregated.items():
            f.write(f"{k}: {v:.4f}\n")

if __name__ == "__main__":
    main()
