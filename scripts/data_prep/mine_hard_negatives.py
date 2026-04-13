
import os
import sys
import torch
import faiss
import json
import ir_datasets
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.append(".")

from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder
from transformers import AutoTokenizer

def mine_hard_negatives(
    checkpoint_path,
    output_path,
    batch_size=512, 
    top_k=200, 
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"Loading model from {checkpoint_path}...")
    # Load onto CPU first
    model = HypencoderDualEncoder.from_pretrained(checkpoint_path)
    
    # DataParallel Setup
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    # 1. Load Corpus & Build Lookup
    print("Loading Corups (MSMARCO) into RAM...")
    ds = ir_datasets.load("msmarco-passage")
    
    doc_id_to_text = {}
    doc_lookup = [] # Map index -> doc_id
    
    # Store text in memory (approx 4GB)
    for doc in tqdm(ds.docs_iter(), desc="Loading Docs"):
        doc_id_to_text[doc.doc_id] = doc.text
        doc_lookup.append(doc.doc_id)
        
    print(f"Loaded {len(doc_lookup)} documents.")

    # 2. Indexing
    index = faiss.IndexFlatIP(768)
    if device == "cuda":
        res = faiss.StandardGpuResources()
        # Use all available GPUs for FAISS
        if torch.cuda.device_count() > 1:
             print("Sharding FAISS index across all GPUs...")
             co = faiss.GpuMultipleClonerOptions()
             co.shard = True # Force sharding to reduce memory duplication
             index = faiss.index_cpu_to_all_gpus(index, co=co)
        else:
             index = faiss.index_cpu_to_gpu(res, 0, index)
    
    # Parallel Encoding Strategy
    passage_encoder = model.module.passage_encoder if isinstance(model, torch.nn.DataParallel) else model.passage_encoder
    
    if torch.cuda.device_count() > 1 and not isinstance(passage_encoder, torch.nn.DataParallel):
         passage_encoder = torch.nn.DataParallel(passage_encoder)

    def batch_encode_optimized(texts, is_query=False):
        all_embs = []
        # Chunk size needs to be large enough to saturate multiple GPUs
        chunk_size = 1024 * torch.cuda.device_count() if torch.cuda.is_available() else 256
        
        for i in range(0, len(texts), chunk_size):
            batch_texts = texts[i:i+chunk_size]
            inputs = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=64 if is_query else 180
            ).to(device)
            
            with torch.no_grad():
                out = passage_encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
                
                # Handle return type safely
                if isinstance(out, (list, tuple)):
                     embs = out[0]
                elif hasattr(out, 'representation'):
                     embs = out.representation
                else:
                     embs = out
            
            all_embs.append(embs.cpu().numpy())
        return np.concatenate(all_embs, axis=0)

    print("Encoding & Indexing Corpus...")
    encode_batch_size = 500000 
    for i in tqdm(range(0, len(doc_lookup), encode_batch_size), desc="Indexing"):
        batch_ids = doc_lookup[i:i+encode_batch_size]
        batch_texts = [doc_id_to_text[did] for did in batch_ids]
        embs = batch_encode_optimized(batch_texts, is_query=False)
        index.add(embs)

    print(f"Index Size: {index.ntotal}")

    # 3. Retrieve for Train Data
    train_file = "./data/triples.train.jsonl"
    print(f"Reading Queries from {train_file}...")
    
    def read_batches(file_path, batch_size):
        batch_triples = []
        batch_queries = []
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    query = item.get('query')
                    if isinstance(query, dict):
                        query = query.get('content') or query.get('text')
                    if not isinstance(query, str) or not query.strip():
                        continue
                        
                    batch_triples.append(item)
                    batch_queries.append(query)
                    
                    if len(batch_triples) >= batch_size:
                        yield batch_triples, batch_queries
                        batch_triples = []
                        batch_queries = []
                except json.JSONDecodeError:
                    continue
            if batch_triples:
                yield batch_triples, batch_queries

    print(f"Starting Mining (Streaming)...")
    
    with open(output_path, 'w') as f_out:
        batch_iterator = read_batches(train_file, batch_size)
        
        for batch_triples, batch_queries in tqdm(batch_iterator, desc="Mining Batches"):
             
            q_embs = batch_encode_optimized(batch_queries, is_query=True)
            D, I = index.search(q_embs, top_k)
            
            for j, (scores, indices) in enumerate(zip(D, I)):
                triple = batch_triples[j]
                current_positives = set(triple.get('positives', []))
                
                hard_negatives = []
                for idx in indices:
                    if idx == -1: continue
                    retrieved_id = doc_lookup[idx]
                    retrieved_text = doc_id_to_text[retrieved_id]
                    
                    if retrieved_text in current_positives:
                        continue
                        
                    hard_negatives.append(retrieved_text)
                
                triple['negatives'] = hard_negatives[:100]
                f_out.write(json.dumps(triple) + "\n")
                
            f_out.flush()

    print(f"Done! Saved to {output_path}")

if __name__ == "__main__":
    import fire
    fire.Fire(mine_hard_negatives)
