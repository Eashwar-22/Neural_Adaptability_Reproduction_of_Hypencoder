
import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from tqdm import tqdm
from transformers import AutoTokenizer

from hypencoder_cb.modeling.hypencoder import Hypencoder

def load_queries(file_path, num_queries=1000):
    lines_to_read = num_queries * 5
    raw_queries = []
    print(f"Sampling queries from {file_path}...")
    with open(file_path, 'r') as f:
        for line in f:
            if len(raw_queries) >= lines_to_read:
                break
            try:
                data = json.loads(line)
                if 'query' in data:
                    q = data['query']
                    content = q if isinstance(q, str) else q.get('content', '')
                    if content.strip():
                        raw_queries.append(content)
            except:
                continue
    
    # Random Sample
    import random
    random.seed(42)
    if len(raw_queries) > num_queries:
        queries = random.sample(raw_queries, num_queries)
    else:
        queries = raw_queries
        
    print(f"Final sample size: {len(queries)} (Balanced from pool of {len(raw_queries)})")
    return queries


# Monkey-patch function generator
def create_probe_hook(captured_weights_list):
    def probe_hook(self, last_hidden_state, attention_mask):
       raise NotImplementedError("This hook is intended to be used with a saved original method")
    return probe_hook

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/triples.train.jsonl")
    parser.add_argument("--output_dir", type=str, default="docs/analysis/umap")
    parser.add_argument("--num_queries", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Model
    print(f"Loading model from {args.model_path}...")
    from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder
    
    # Load the full dual encoder
    model = HypencoderDualEncoder.from_pretrained(args.model_path)
    
    #  query encoder
    query_encoder = model.query_encoder
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Monkey-patch the query_encoder instance
    captured_weights = []
    
    # Save original method
    original_get_weights = query_encoder._get_weights_and_biases
    
    # Define hook
    def get_weights_hook(last_hidden_state, attention_mask):
        weights, biases, reg_loss = original_get_weights(last_hidden_state, attention_mask)
        # Capture
        w_detached = [w.detach().cpu() for w in weights]
        captured_weights.append(w_detached)
        return weights, biases, reg_loss
    
    # Apply patch
    query_encoder._get_weights_and_biases = get_weights_hook

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except Exception as e:
        print(f"Failed to load tokenizer from {args.model_path}, falling back to bert-base-uncased. Error: {e}")
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    # 2. Load Data
    queries = load_queries(args.data_path, num_queries=args.num_queries)

    # 3. Generate Weights
    print("Generating Q-Net weights...")
    
    all_weights_by_layer = {} # layer_idx -> list of tensors (batch, flattened_dim)

    # Process in batches
    for i in tqdm(range(0, len(queries), args.batch_size)):
        batch_queries = queries[i : i + args.batch_size]
        inputs = tokenizer(batch_queries, return_tensors="pt", padding=True, truncation=True, max_length=64)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # We only need to run the query encoder
            query_encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    
    # 3. Step 1: Extract Final Layer and Generate Clustered Labels
    print("\nPhase 1: Generating Clustered Labels from Final Layer (L6)...")
    final_layer_weights = []
    for i in tqdm(range(0, len(queries), args.batch_size), desc="Extracting Final Layer"):
        batch_queries = queries[i : i + args.batch_size]
        inputs = tokenizer(batch_queries, return_tensors="pt", padding=True, truncation=True, max_length=64)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        captured_weights.clear()
        with torch.no_grad():
            query_encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        
        if captured_weights:
            batch_layer_tensors = captured_weights[0]
            w_tensor = batch_layer_tensors[-1].detach() # Layer 6
            flat_cpu = w_tensor.reshape(w_tensor.size(0), -1).to(torch.float16).cpu().numpy()
            final_layer_weights.append(flat_cpu)

    final_layer_weights = np.concatenate(final_layer_weights, axis=0)
    
    # L2 Normalization (CRITICAL for Cosine consistency)
    from sklearn.preprocessing import normalize
    final_layer_weights_norm = normalize(final_layer_weights.astype(np.float32), axis=1)
    
    # PCA to 100 dims
    from sklearn.decomposition import PCA
    pca_final = PCA(n_components=100, random_state=42)
    final_reduced = pca_final.fit_transform(final_layer_weights_norm)
    
    # UMAP for Final Layer
    print("  Fitting UMAP for Final Layer...")
    reducer_final = UMAP(n_neighbors=50, min_dist=0.0, metric='cosine', random_state=42)
    embedding_final = reducer_final.fit_transform(final_reduced)
    
    # Cluster on the 2D UMAP Embedding
    # This ensures that the groups the user sees in 2D are actually the clusters.
    from sklearn.cluster import KMeans
    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embedding_final)
    
    print(f"Assigned {num_clusters} clusters based on Layer 6 UMAP manifold.")
    
    # 4. Phase 2: Generate all layers using these labels
    print("\nPhase 2: Generating UMAPs for all layers...")
    sns.set_theme(style="white")
    palette = sns.color_palette("tab10", n_colors=num_clusters)
    
    num_layers = 7
    fig, axes = plt.subplots(4, 2, figsize=(12, 24))
    axes = axes.flatten()

    # Pre-store embedding_final for the loop
    umap_embeddings = {}
    umap_embeddings[6] = embedding_final
    
    for l_idx in range(num_layers):
        print(f"\nProcessing Layer {l_idx}...")
        
        if l_idx == 6:
            embedding = embedding_final
        else:
            layer_weights = []
            for i in tqdm(range(0, len(queries), args.batch_size), desc=f"Extracting L{l_idx}"):
                batch_queries = queries[i : i + args.batch_size]
                inputs = tokenizer(batch_queries, return_tensors="pt", padding=True, truncation=True, max_length=64)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                captured_weights.clear()
                with torch.no_grad():
                    query_encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                
                if captured_weights:
                    batch_layer_tensors = captured_weights[0]
                    if l_idx < len(batch_layer_tensors):
                        w_tensor = batch_layer_tensors[l_idx].detach()
                        flat_cpu = w_tensor.reshape(w_tensor.size(0), -1).to(torch.float16).cpu().numpy()
                        layer_weights.append(flat_cpu)
            
            layer_weights = np.concatenate(layer_weights, axis=0)
            layer_weights_norm = normalize(layer_weights.astype(np.float32), axis=1)
            
            pca = PCA(n_components=100, random_state=42)
            layer_reduced = pca.fit_transform(layer_weights_norm)
            del layer_weights, layer_weights_norm
            
            reducer = UMAP(n_neighbors=50, min_dist=0.0, metric='cosine', random_state=42)
            embedding = reducer.fit_transform(layer_reduced)
            del layer_reduced

        # Plot in Grid
        ax = axes[l_idx]
        for c in range(num_clusters):
            mask = cluster_labels == c
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=8, alpha=0.6, 
                       color=palette[c], label=f"Archetype {c+1}")
        ax.set_title(f"Q-Net Layer {l_idx}", fontsize=14, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc='best', markerscale=1.5, fontsize=8)

        # Individual Plot (High-Res)
        plt.figure(figsize=(10, 8))
        ax_ind = plt.gca()
        for c in range(num_clusters):
            mask = cluster_labels == c
            plt.scatter(embedding[mask, 0], embedding[mask, 1], s=12, alpha=0.7, 
                        color=palette[c], label=f"Cluster {c+1}")
        
        # Add Title in top-left of the plot area
        # (Removed as per user feedback: looks ugly)
        
        if l_idx == 6:
            plt.legend(loc='best', markerscale=2, title="Clusters", fontsize=10)
        
        plt.xticks([]); plt.yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"qnet_umap_layer_{l_idx}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # Save consolidated page
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "qnet_umap_dedicated_page.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nAll plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
