import os
import argparse
import torch
import ir_datasets
import pytrec_eval
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
from typing import List, Dict

# Reuse Hypencoder metric calculation if possible, or reimplement standard metrics
# We will reimplement standard metrics to be standalone but compatible format

def load_bge_model(model_name: str = "BAAI/bge-base-en-v1.5", device: str = "cuda"):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)
    return model, tokenizer

def encode(texts: List[str], model, tokenizer, batch_size=256, max_length=512, device="cuda", desc="Encoding"):
    embeddings = []
    
    # Sort/batching optimization could be done, but keeping it simple
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = texts[i : i + batch_size]
        
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # CLS pooling for BGE
            # Note: BGE uses CLS token normalized
            cls_embeddings = outputs.last_hidden_state[:, 0]
            cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)
            embeddings.append(cls_embeddings.cpu())
            
    return torch.cat(embeddings, dim=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--ir_dataset_name", type=str, required=True, help="e.g. beir/nfcorpus")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--query_max_length", type=int, default=64)
    parser.add_argument("--doc_max_length", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=1000)
    args = parser.parse_args()

    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Dataset
    print(f"Loading dataset: {args.ir_dataset_name}")
    dataset = ir_datasets.load(args.ir_dataset_name)
    
    # Pre-load data to handle iterators
    docs_iter = dataset.docs_iter()
    queries_iter = dataset.queries_iter()
    
    doc_ids = []
    doc_texts = []
    # Adjust field names based on dataset schema if needed, but usually 'text' or 'title'+'text'
    # BGE expects "Represent this document for retrieval: " instruction? 
    # BGE v1.5 instruction: "Represent this sentence for searching relevant passages: " for QUERIES
    # Documents usually don't need instruction, just title+text
    
    print("Reading documents...")
    for doc in tqdm(docs_iter):
        doc_ids.append(doc.doc_id)
        # Simple concat of title + text
        text = (getattr(doc, "title", "") + " " + getattr(doc, "text", "")).strip()
        doc_texts.append(text)
        
    query_ids = []
    query_texts = []
    print("Reading queries...")
    for query in tqdm(queries_iter):
        query_ids.append(query.query_id)
        # ADD INSTRUCTION FOR BGE QUERY
        instruction = "Represent this sentence for searching relevant passages: "
        query_texts.append(instruction + query.text)

    # Load Model
    model, tokenizer = load_bge_model(args.model_name, args.device)
    
    # Encode Corpus
    print("Encoding documents...")
    if os.path.exists(output_dir / "doc_embeddings.pt"):
        doc_embeddings = torch.load(output_dir / "doc_embeddings.pt")
        print("Loaded cached doc embeddings.")
    else:
        doc_embeddings = encode(
            doc_texts, model, tokenizer, 
            batch_size=args.batch_size, 
            max_length=args.doc_max_length, 
            device=args.device,
            desc="Encode Docs"
        )
        torch.save(doc_embeddings, output_dir / "doc_embeddings.pt")
        
    # Encode Queries
    print("Encoding queries...")
    query_embeddings = encode(
        query_texts, model, tokenizer, 
        batch_size=args.batch_size, 
        max_length=args.query_max_length, 
        device=args.device,
        desc="Encode Queries"
    )
    
    # Retrieval (Dot Product / Cosine Similarity)
    # Since embeddings are normalized, Dot Product == Cosine Similarity
    print("Running retrieval...")
    
    # Move huge doc matrix to GPU in chunks if needed, or all at once if memory allows
    # 3k docs (NFCorpus) fits easily. 8.8M (MS MARCO) does NOT.
    # For this script (assuming BEIR datasets < 1M usually), we might try all gpu.
    # But to be safe, we iterate.
    
    results = {}
    
    doc_embeddings = doc_embeddings.to(args.device)
    query_embeddings = query_embeddings.to(args.device)
    
    # Block matrix multiplication
    # Q x D^T
    # Split queries into chunks to avoid OOM
    
    chunk_size = 1000
    num_queries = len(query_texts)
    
    with torch.no_grad():
        for i in range(0, num_queries, chunk_size):
            q_chunk = query_embeddings[i : i + chunk_size] # (Batch, Dim)
            scores = torch.mm(q_chunk, doc_embeddings.t()) # (Batch, NumDocs)
            
            # Top-k
            topk_scores, topk_indices = torch.topk(scores, k=min(args.top_k, len(doc_texts)), dim=1)
            
            topk_scores = topk_scores.cpu().numpy()
            topk_indices = topk_indices.cpu().numpy()
            
            for j, (sc, idx) in enumerate(zip(topk_scores, topk_indices)):
                qid = query_ids[i + j]
                results[qid] = {doc_ids[k]: float(v) for k, v in zip(idx, sc)}

    # Save Run File
    print("Saving run file...")
    run_file = output_dir / "run.json"
    import json
    with open(run_file, "w") as f:
        json.dump(results, f)
        
    # Standard Trec Run Format
    trec_file = output_dir / "run.trec"
    with open(trec_file, "w") as f:
        for qid, docs in results.items():
            rank = 1
            # Sort just in case
            sorted_docs = sorted(docs.items(), key=lambda item: item[1], reverse=True)
            for docid, score in sorted_docs:
                f.write(f"{qid} Q0 {docid} {rank} {score:.4f} BGE-Base\n")
                rank += 1
                
    # Evaluation
    print("Evaluating...")
    qrels = {}
    try:
        if hasattr(dataset, 'qrels_iter'):
            for qrel in dataset.qrels_iter():
                if qrel.query_id not in qrels:
                    qrels[qrel.query_id] = {}
                qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
        else:
            print(f"Dataset {args.ir_dataset_name} does not have qrels_iter. Skipping internal evaluation.")
    except Exception as e:
        print(f"Error loading qrels: {e}")
        print("Skipping internal evaluation.")
    
    if not qrels:
        print("No qrels found. Skipping evaluation metrics calculation.")
        return

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut_10', 'map_cut_1000', 'recip_rank', 'recall_1000'})
    metrics = evaluator.evaluate(results)
    
    # Aggregate
    aggregated = {
        'nDCG@10': np.mean([v['ndcg_cut_10'] for v in metrics.values()]),
        'MRR': np.mean([v['recip_rank'] for v in metrics.values()]),
        'R@1000': np.mean([v['recall_1000'] for v in metrics.values()])
    }
    
    print("================ RESULTS ================")
    for k, v in aggregated.items():
        print(f"{k}: {v:.4f}")
    
    # Save aggregated metrics
    with open(output_dir / "aggregated_metrics.txt", "w") as f:
        for k, v in aggregated.items():
            f.write(f"{k}: {v:.4f}\n")
            
    print(f"Done. Outputs in {output_dir}")

if __name__ == "__main__":
    main()
