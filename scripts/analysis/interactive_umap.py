
import os
import argparse
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from umap import UMAP
import plotly.express as px
from transformers import AutoTokenizer
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder
from collections import defaultdict

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
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_queries", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model_path}...")
    model = HypencoderDualEncoder.from_pretrained(args.model_path)
    model.eval()
    
    if torch.cuda.is_available():
        print("CUDA detected, moving model to GPU.")
        device = torch.device("cuda")
        model.to(device)
    else:
        print("WARNING: CUDA not detected, running on CPU.")
        device = torch.device("cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except:
        print("Tokenizer not found, using default.")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    print(f"Loading queries from {args.data_path}...")
    queries = load_queries(args.data_path, args.num_queries)
    print(f"Loaded {len(queries)} queries.")

    # --- Weights Capture ---
    all_weights_by_layer = defaultdict(list)
    original_get_weights = model.query_encoder._get_weights_and_biases

    print("Monkey patch applied.")
    def captured_get_weights(last_hidden_state, attention_mask):
        weights_list, biases_list, reg_loss = original_get_weights(last_hidden_state, attention_mask)
        
        # FIX: Iterate over weights only. 
        # Biases have len 6, Weights have len 7. Zip was truncating Layer 6.
        # Also, static plots ignored biases, so we should too for replication.
        for i, W in enumerate(weights_list):
            batch_size = W.shape[0]
            curr_W = W.detach().cpu().numpy()
            
            # Flatten [batch, in, out] -> [batch, in*out]
            flat_W = curr_W.reshape(batch_size, -1)
            
            # append only weights
            all_weights_by_layer[i].append(flat_W)
            
        return weights_list, biases_list, reg_loss

    model.query_encoder._get_weights_and_biases = captured_get_weights

    batch_size = 32
    print("Generating Q-Net weights...")
    for i in tqdm(range(0, len(queries), batch_size)):
        batch_queries = queries[i : i + batch_size]
        inputs = tokenizer(batch_queries, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            model.query_encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    num_layers = max(all_weights_by_layer.keys()) + 1
    print(f"Captured weights for {num_layers} layers.")
    
    for i in range(num_layers):
        all_weights_by_layer[i] = np.concatenate(all_weights_by_layer[i], axis=0)

    # --- Clustering (Track Routing) ---
    # We use PCA specifically for stable Clustering, but use Raw Data for UMAP (to match static plots)
    print(f"\nRunning K-Means on Final Layer ({num_layers-1})...")
    
    # PCA just for Clustering efficiency/stability
    pca_for_clustering = PCA(n_components=50, random_state=42)
    layer_final_raw = all_weights_by_layer[num_layers-1]
    layer_final_pca = pca_for_clustering.fit_transform(layer_final_raw)
    
    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(layer_final_pca)
    cluster_names = [f"Cluster {c}" for c in cluster_labels]
    
    print("  Clustering done.")

    # --- Unified UMAP & Plotting ---
    print("\nRunning Unified UMAP & Generating Animation...")
    print("NOTE: Using Raw Weights + Tight Parameters (n_neighbors=30, min_dist=0.0) to match static plots.")
    
    all_dfs = []
    
    # Pre-calculate embeddings for all layers
    for layer_idx in range(num_layers):
        print(f"  UMAP on Layer {layer_idx} (Raw Data)...")
        raw_data = all_weights_by_layer[layer_idx]
        
        # EXACT PARAMETERS from static individual plots
        umap_reducer = UMAP(n_neighbors=30, min_dist=0.0, metric='cosine', random_state=42)
        embedding = umap_reducer.fit_transform(raw_data)
        
        df_layer = pd.DataFrame({
            'x': embedding[:, 0],
            'y': embedding[:, 1],
            'query': queries,
            'cluster': cluster_names,
            'layer': layer_idx  # Animation Frame Key
        })
        all_dfs.append(df_layer)

        # DEBUG: Save static PNG to compare with user's reference
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 8))
            plt.scatter(embedding[:, 0], embedding[:, 1], s=10, alpha=0.6, c='teal')
            plt.title(f"Layer {layer_idx} UMAP (Cosine)")
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f"debug_layer_{layer_idx}.png"))
            plt.close()
        except ImportError:
            pass

    # Combine
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # Calculate global range for stable animation axes
    x_min, x_max = full_df['x'].min(), full_df['x'].max()
    y_min, y_max = full_df['y'].min(), full_df['y'].max()
    
    # Padding
    padding = 1.0
    range_x = [x_min - padding, x_max + padding]
    range_y = [y_min - padding, y_max + padding]
    
    print("  Generating Unified HTML...")
    fig = px.scatter(
        full_df,
        x='x',
        y='y',
        animation_frame='layer',  # THE MAGIC SLIDER
        animation_group='query',  # Match points across frames (by query text or index?) Index is safer implicitly.
        color='cluster',
        hover_data=['query'],
        range_x=range_x,
        range_y=range_y,
        title='Hypencoder Q-Net Weights: Evolution from Layer 0 to Final',
        template='plotly_dark',
        width=1000,
        height=800
    )
    
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000 # Speed of auto-play
    
    out_file = os.path.join(args.output_dir, "interactive_all_layers.html")
    fig.write_html(out_file)
    print(f"Done! Saved to {out_file}")

if __name__ == "__main__":
    main()
