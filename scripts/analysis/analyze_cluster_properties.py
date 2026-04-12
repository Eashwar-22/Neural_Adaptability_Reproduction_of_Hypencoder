
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
    
    # Capture Layer 6
    captured_weights = []
    def hook(hidden, mask):
        w, b, reg = model.query_encoder._get_weights_and_biases(hidden, mask)
        captured_weights.append(w[6].detach().cpu())
        return w, b, reg
    
    # Monkey patch ONLY for running forward pass
    # We need to preserve the original method to call it inside hook? 
    # The hook above calls `model.query_encoder._get_weights_and_biases`... RECURSION ERROR!
    # Fix: Save original reference.
    
    original_method = model.query_encoder._get_weights_and_biases
    
    # Define safe hook
    def safe_hook(hidden, mask):
        w, b, reg = original_method(hidden, mask)
        captured_weights.append(w[6].detach().cpu())
        return w, b, reg
        
    model.query_encoder._get_weights_and_biases = safe_hook

    print("Generating weights...")
    batch_size = 32
    for i in tqdm(range(0, len(queries), batch_size)):
        batch_queries = queries[i : i + batch_size]
        inputs = tokenizer(batch_queries, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            model.query_encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    final_layer_data = []
    for w_batch in captured_weights:
        flattened = w_batch.reshape(w_batch.shape[0], -1)
        final_layer_data.append(flattened)
    final_layer_np = torch.cat(final_layer_data, dim=0).numpy()
    
    print("Clustering...")
    pca = PCA(n_components=50, random_state=42)
    reduced = pca.fit_transform(final_layer_np)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = kmeans.fit_predict(reduced)
    
    # --- PHYSICAL ANALYSIS ---
    print("\n=== Cluster Physical Analysis ===")
    
    clusters = defaultdict(list)
    for q, label in zip(queries, labels):
        clusters[label].append(q)
        
    for c_id in sorted(clusters.keys()):
        qs = clusters[c_id]
        
        # 1. Length (Chars)
        lengths = [len(q) for q in qs]
        avg_len = np.mean(lengths)
        
        # 2. Token Count
        token_counts = [len(tokenizer.tokenize(q)) for q in qs]
        avg_tokens = np.mean(token_counts)
        
        # 3. Start Words (First token)
        start_words = [q.split()[0].lower() if q.split() else "" for q in qs]
        common_starts = Counter(start_words).most_common(3)
        
        # 4. Question Words?
        wh_words = ['who', 'what', 'where', 'when', 'why', 'how', 'is', 'can', 'does']
        has_wh = sum(1 for w in start_words if w in wh_words)
        wh_ratio = has_wh / len(qs)

        print(f"\nCluster {c_id} (N={len(qs)}):")
        print(f"  Avg Length (chars): {avg_len:.1f}")
        print(f"  Avg Tokens (bert):  {avg_tokens:.1f}")
        print(f"  Top Start Words:    {common_starts}")
        print(f"  'WH-Question' %:    {wh_ratio*100:.1f}%")

if __name__ == "__main__":
    main()
