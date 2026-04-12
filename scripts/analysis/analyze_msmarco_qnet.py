
import os
import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder

def load_random_queries_from_jsonl(filepath, num_samples=200, search_space=10000):
    """
    Reads the first `search_space` lines from the jsonl file and samples `num_samples` queries.
    """
    print(f"Sampling {num_samples} queries from the first {search_space} lines of {filepath}...")
    queries = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i >= search_space:
                break
            try:
                data = json.loads(line)
                # Structure: {"query": {"content": "text", ...}, ...}
                query_text = data.get("query", {}).get("content", "")
                if query_text:
                    queries.append(query_text)
            except json.JSONDecodeError:
                continue
    
    if len(queries) < num_samples:
        print(f"Warning: Only found {len(queries)} valid queries. Using all of them.")
        return queries
    
    return random.sample(queries, num_samples)

def compute_metrics(q_net_layer0_weight):
    """
    Computes RSD and Spectral Entropy for a single weight matrix.
    Input shape: (Out, In) -> (768, 768)
    """
    # 1. RSD (Relative Standard Deviation)
    # Measure of adaptation magnitude
    mean_abs = torch.mean(torch.abs(q_net_layer0_weight))
    param_std = torch.std(q_net_layer0_weight, dim=0) # Std across input dim? No, across... wait.
    # In depth analysis: get_rsd took a tensor.
    # Here we have a SINGLE matrix for a single query.
    # Wait, RSD is usually defined across a BATCH of queries to see how much it varies PER PARAMETER.
    # "Experiment B: Layer Depth Analysis ... Metric: Relative Standard Deviation (RSD) across the query batch."
    # So I cannot compute RSD for a *single* query. I need the batch of generated weights.
    return {} # Placeholder, actual logic in main loop

def analyze_batch_properties(weights_tensor):
    """
    weights_tensor: (Batch, Out, In) -> (N, 768, 768)
    Computes metrics across the batch.
    """
    # 1. RSD per parameter, averaged
    # Std dev of each parameter across the batch
    param_std = torch.std(weights_tensor, dim=0) # (Out, In)
    mean_param_std = torch.mean(param_std)
    
    # Mean magnitude of parameters
    mean_abs = torch.mean(torch.abs(weights_tensor))
    
    rsd = (mean_param_std / mean_abs).item() * 100
    return rsd

def compute_spectral_entropy(weight_matrix):
    """
    Computes spectral entropy of a single matrix (Out, In).
    """
    # Singular Values
    try:
        _, S, _ = torch.svd(weight_matrix)
        # Normalize
        S_norm = S / (torch.sum(S) + 1e-9)
        # Entropy
        entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-9)).item()
        return entropy
    except Exception as e:
        print(f"SVD failed: {e}")
        return 0.0

def main(num_samples=200):
    # Settings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = "data/triples.train.jsonl"
    checkpoint_path = "./checkpoints/hypencoder.6_layer_full_real_opt"
    output_dir = "docs/analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    queries = load_random_queries_from_jsonl(data_path, num_samples=num_samples)
    print(f"Loaded {len(queries)} queries. Examples: {queries[:3]}")
    
    # 2. Load Model
    print(f"Loading model from {checkpoint_path}...")
    model_dual = HypencoderDualEncoder.from_pretrained(checkpoint_path, use_safetensors=True).to(device)
    model = model_dual.query_encoder
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model.eval()
    
    # 3. Generate Q-Nets
    print("Generating Q-Nets...")
    # Process in batches to avoid OOM if N is large, but for 200 it fits in memory easily.
    
    inputs = tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        output = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        q_net = output.representation
        # Layer 0 weights: (Batch, 768, 768)
        # Access: q_net.layers[0].linear.weight
        # Note: q_net is a NoTorchSequential, elements are NoTorchDenseBlock or NoTorchLinear
        # Checking QNet_depth_analysis.py: q_net.layers[0].linear.weight
        layer0_weights = q_net.layers[0].linear.weight
    
    print(f"Generated weights shape: {layer0_weights.shape}")
    
    # 4. Compute Metrics
    
    # A. RSD (Global)
    rsd = analyze_batch_properties(layer0_weights)
    print(f"Global Layer 0 RSD (Adaptation Rate): {rsd:.2f}%")
    
    # B. Per-Query Spectral Complexity
    entropies = []
    
    print("Computing Spectral Entropies...")
    for i in range(len(queries)):
        w = layer0_weights[i]
        ent = compute_spectral_entropy(w)
        entropies.append(ent)
        
    entropies = np.array(entropies)
    
    # 5. Analysis & Visualization
    
    # Query Length Analysis
    lengths = [len(q.split()) for q in queries]
    
    # Stats
    # Identify Top/Bottom Entropy Queries
    # High Entropy = "Simple/Identity-like" (Uniform singular values? No wait.)
    # Low Entropy = "Concentrated/Low Rank" (Few dominant singular values)
    # In QNet_depth_analysis: "Specific queries 'collapse' dimensions... -> Lower Effective Rank (Entropy)"
    # "Simple Entities -> Higher Effective Rank"
    
    top_k = 5
    sorted_indices = np.argsort(entropies)
    
    print("\n--- Queries with LOWEST Entropy (Most Specialized/Collapsed) ---")
    for i in sorted_indices[:top_k]:
        print(f"Entropy: {entropies[i]:.4f} | Query: {queries[i]}")
        
    print("\n--- Queries with HIGHEST Entropy (Most Broad/Complex) ---")
    for i in sorted_indices[-top_k:]:
        print(f"Entropy: {entropies[i]:.4f} | Query: {queries[i]}")

    # Plot 1: Entropy (Complexity) Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(entropies, bins=20, kde=True)
    plt.title(f"Distribution of Q-Net Spectral Entropy (n={len(queries)})")
    plt.xlabel("Spectral Entropy (Effective Rank)")
    plt.ylabel("Count")
    plt.axvline(x=np.mean(entropies), color='r', linestyle='--', label=f"Mean: {np.mean(entropies):.2f}")
    plt.legend()
    plt.savefig(f"{output_dir}/msmarco_spectral_complexity.png")
    print(f"Saved {output_dir}/msmarco_spectral_complexity.png")
    
    # Plot 2: Entropy vs Query Length
    plt.figure(figsize=(10, 6))
    plt.scatter(lengths, entropies, alpha=0.6)
    plt.title("Query Length vs Spectral Complexity")
    plt.xlabel("Query Word Count")
    plt.ylabel("Spectral Entropy")
    
    # Add trendline
    z = np.polyfit(lengths, entropies, 1)
    p = np.poly1d(z)
    plt.plot(lengths, p(lengths), "r--", label=f"Trend: slope={z[0]:.4f}")
    plt.legend()
    
    plt.savefig(f"{output_dir}/msmarco_entropy_vs_length.png")
    print(f"Saved {output_dir}/msmarco_entropy_vs_length.png")
    
    # Save Report Data
    with open(f"{output_dir}/msmarco_analysis_summary.json", "w") as f:
        json.dump({
            "global_rsd": rsd,
            "mean_entropy": float(np.mean(entropies)),
            "std_entropy": float(np.std(entropies)),
            "lowest_entropy_queries": [{"q": queries[i], "e": float(entropies[i])} for i in sorted_indices[:top_k]],
            "highest_entropy_queries": [{"q": queries[i], "e": float(entropies[i])} for i in sorted_indices[-top_k:]]
        }, f, indent=2)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
