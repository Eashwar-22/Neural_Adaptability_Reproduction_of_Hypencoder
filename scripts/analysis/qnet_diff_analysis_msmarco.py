
import os
import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from transformers import AutoTokenizer
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder

def load_random_queries_from_jsonl(filepath, num_samples=50, search_space=5000):
    """
    Reads queries from jsonl.
    """
    print(f"Sampling {num_samples} queries from the first {search_space} lines of {filepath}...")
    queries = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i >= search_space:
                break
            try:
                data = json.loads(line)
                query_text = data.get("query", {}).get("content", "")
                if query_text:
                    queries.append(query_text)
            except json.JSONDecodeError:
                continue
    
    if len(queries) < num_samples:
        return queries
    
    return random.sample(queries, num_samples)

def compute_pairwise_distances(tensors, metric='euclidean'):
    """
    Computes pairwise distance matrix.
    tensors: (Batch, Dim)
    """
    n = tensors.shape[0]
    dist_matrix = np.zeros((n, n))
    
    # We can use broadcasting or simple loops. For 50 samples, loop is fine and clear.
    # But vectorized is better for larger dims.
    t = tensors.cpu()
    if metric == 'euclidean':
        # ||A - B||_2
        dist_matrix = torch.cdist(t, t, p=2).numpy()
    elif metric == 'manhattan':
        # ||A - B||_1
        dist_matrix = torch.cdist(t, t, p=1).numpy()
        
    return dist_matrix

def main():
    # Settings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = "data/triples.train.jsonl" # Expecting to be run from project root
    checkpoint_path = "./checkpoints/hypencoder.6_layer_full_real_opt"
    output_dir = "docs/analysis/diff_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    # Use 32 queries for a clean heatmap
    queries = load_random_queries_from_jsonl(data_path, num_samples=32)
    print(f"Analyzing {len(queries)} queries...")
    
    # 2. Load Model
    print(f"Loading model from {checkpoint_path}...")
    model_dual = HypencoderDualEncoder.from_pretrained(checkpoint_path, use_safetensors=True).to(device)
    model = model_dual.query_encoder
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model.eval()
    
    # 3. Generate Weights
    inputs = tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        output = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        q_net = output.representation
        
        # Layer 0 (Input Projection)
        l0_weight = q_net.layers[0].linear.weight # (Batch, 768, 768)
        w_0_flat = l0_weight.reshape(l0_weight.shape[0], -1)
        
        # Final Layer (Scoring)
        l_last_weight = q_net.layers[-1].weight # (Batch, 768, 1)
        w_last_flat = l_last_weight.reshape(l_last_weight.shape[0], -1)

    print(f"Layer 0 Weights Shape: {w_0_flat.shape}")
    print(f"Last Layer Weights Shape: {w_last_flat.shape}")
    
    # 4. Compute Pairwise Differences (Euclidean Distance)
    print("Computing pairwise Euclidean distances...")
    dist_0 = compute_pairwise_distances(w_0_flat, metric='euclidean')
    dist_last = compute_pairwise_distances(w_last_flat, metric='euclidean')
    
    # 5. Visualization
    
    def plot_heatmap(matrix, query_labels, title, filename):
        plt.figure(figsize=(14, 12))
        # Shorten labels
        short_labels = [q[:30] + "..." if len(q) > 30 else q for q in query_labels]
        
        sns.heatmap(matrix, xticklabels=short_labels, yticklabels=short_labels, cmap="viridis", annot=False)
        plt.title(title)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Saved {filename}")

    # Plot Raw Distances
    plot_heatmap(dist_0, queries, "Layer 0 Pairwise Euclidean Distance (Input Projection)", f"{output_dir}/dist_layer0.png")
    plot_heatmap(dist_last, queries, "Last Layer Pairwise Euclidean Distance (Scoring)", f"{output_dir}/dist_last.png")

    # 6. Residual Analysis (Difference from Mean)
    # The user asked for "take the differences instead". 
    # Usually this means how different are they from each other? Which is pairwise distance.
    # But let's also look at magnitudes of difference from MEAN.
    
    w_mean = torch.mean(w_last_flat, dim=0, keepdim=True)
    w_residuals = w_last_flat - w_mean
    
    # To visualize "element wise difference" directly for 32 queries is hard (32 x 768 matrix).
    # But we can plot the heatmap of the Residuals Matrix itself? 
    # i.e. x-axis = parameter index, y-axis = query.
    
    plt.figure(figsize=(15, 8))
    # We can't plot 768 columns easily. Let's look at the top 50 most variable parameters.
    param_var = torch.var(w_last_flat, dim=0).cpu().numpy()
    top_indices = np.argsort(param_var)[-50:]
    
    res_numpy = w_residuals.cpu().numpy()
    res_subset = res_numpy[:, top_indices]
    
    sns.heatmap(res_subset, yticklabels=[q[:20] for q in queries], cmap="coolwarm", center=0)
    plt.title("Element-wise Residuals (Top 50 Most Variable Params)")
    plt.xlabel("Parameter Index")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/elementwise_residuals_heatmap.png")
    print(f"Saved {output_dir}/elementwise_residuals_heatmap.png")

    # Save summary stats
    mean_dist_0 = np.mean(dist_0)
    mean_dist_last = np.mean(dist_last)
    
    print(f"\nMean Eucl Distance Layer 0: {mean_dist_0:.4f}")
    print(f"Mean Eucl Distance Last Layer: {mean_dist_last:.4f}")
    
    with open(f"{output_dir}/stats.txt", "w") as f:
        f.write(f"Mean Eucl Distance Layer 0: {mean_dist_0}\n")
        f.write(f"Mean Eucl Distance Last Layer: {mean_dist_last}\n")

if __name__ == "__main__":
    main()
