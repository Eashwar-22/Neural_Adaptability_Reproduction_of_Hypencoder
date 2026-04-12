
import os
import argparse
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from transformers import AutoTokenizer
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder
from collections import defaultdict, Counter

def load_queries(data_path, num_queries=None):
    queries = []
    with open(data_path, 'r') as f:
        for line in f:
            if num_queries and len(queries) >= num_queries:
                break
            item = json.loads(line)
            if 'query' in item:
                q_content = item['query']
                if isinstance(q_content, dict):
                    q_text = q_content.get('content', '') or q_content.get('text', '')
                else:
                    q_text = str(q_content)
                if q_text.strip():
                    queries.append(q_text)
    return queries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_queries", type=int, default=1000)
    args = parser.parse_args()

    # Reuse previous logic to get labels... or just save them next time!
    # Faster: Compute labels again.
    
    print(f"Loading model: {args.model_path}")
    model = HypencoderDualEncoder.from_pretrained(args.model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    queries = load_queries(args.data_path, args.num_queries)
    
    # Capture ALL 7 Layers
    captured_weights = [[] for _ in range(7)]
    
    original_method = model.query_encoder._get_weights_and_biases
    
    def safe_hook(hidden, mask):
        w, b, reg = original_method(hidden, mask)
        # Iterate over all 7 weight layers
        for layer_idx in range(7):
            captured_weights[layer_idx].append(w[layer_idx].detach().cpu())
        return w, b, reg
        
    model.query_encoder._get_weights_and_biases = safe_hook

    print("Generating weights (Forward Pass)...")
    batch_size = 32
    for i in tqdm(range(0, len(queries), batch_size)):
        batch_queries = queries[i : i + batch_size]
        inputs = tokenizer(batch_queries, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            model.query_encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    print("\nAnalyzing Layer Evolution...")
    
    # Analyze each layer independently
    for layer_idx in range(7):
        print(f"\n[[ LAYER {layer_idx} ]]")
        
        # Flatten weights for this layer
        layer_data_list = []
        for w_batch in captured_weights[layer_idx]:
            # w_batch: [batch, 1, 768] usually for weights?
            # Or [batch, num_heads, dim_per_head] -> flattened
            flattened = w_batch.reshape(w_batch.shape[0], -1)
            layer_data_list.append(flattened)
        
        layer_data_np = torch.cat(layer_data_list, dim=0).numpy()
        
        # Cluster This Layer (k=5)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        # Use PCA first for stability?
        # Yes, usually better with high dimensions (768 * heads)
        pca = PCA(n_components=50, random_state=42)
        reduced = pca.fit_transform(layer_data_np)
        
        labels = kmeans.fit_predict(reduced)
        
        # Analyze Themes
        clusters = defaultdict(list)
        for q, label in zip(queries, labels):
            clusters[label].append(q)
            
        # For each cluster, find top start words and common words
        for c_id in sorted(clusters.keys()):
            qs = clusters[c_id]
            
            # Start Words
            start_words = [q.split()[0].lower() if q.split() else "" for q in qs]
            top_starts = Counter(start_words).most_common(3)
            
            # Common Words (excluding stop words simple list)
            all_words = []
            stop_words = {'what', 'how', 'is', 'the', 'a', 'an', 'are', 'in', 'of', 'to', 'for', 'do', 'does', 'did'}
            for q in qs:
                words = [w.lower() for w in q.replace('?', '').split()]
                all_words.extend([w for w in words if w not in stop_words])
            top_words = Counter(all_words).most_common(5)
            
            # Sample Query
            sample = qs[0] if qs else ""
            if len(qs) > 5:
                sample = qs[5] # Pick 5th element
            
            print(f"  Cluster {c_id} (N={len(qs)}):")
            print(f"    Top Starts: {top_starts}")
            print(f"    Top Words:  {top_words}")
            print(f"    Sample:     {sample}")

if __name__ == "__main__":
    main()
