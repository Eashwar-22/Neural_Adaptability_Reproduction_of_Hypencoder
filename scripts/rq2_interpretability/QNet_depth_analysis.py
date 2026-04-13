
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from transformers import AutoTokenizer
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder
from hypencoder_cb.modeling.q_net import NoTorchDenseBlock, NoTorchLinear

def analyze_depth():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Model (Parent Dir + SafeTensors)
    checkpoint_path = "./checkpoints/hypencoder.6_layer_full_real_opt"
    print(f"Loading model from {checkpoint_path}...")
    
    model_dual = HypencoderDualEncoder.from_pretrained(checkpoint_path, use_safetensors=True).to(device)
    model = model_dual.query_encoder
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model.eval()

    # 2. Define Diverse Queries (32 total, 8 archetypes x 4)
    queries_dict = {
        "Medical": ["symptoms of cardiac arrest", "how long do flu symptoms last", "causes of high blood pressure", "treatment for common cold"],
        "Technical": ["how to install python on ubuntu", "center a div in css", "connect to wifi on raspberry pi", "git merge vs rebase"],
        "Entity": ["who is the CEO of apple", "birthplace of albert einstein", "who wrote the great gatsby", "current governor of california"],
        "Factoid": ["distance from earth to moon", "boiling point of water at sea level", "capital of australia", "number of states in the us"],
        "Definition": ["define photosynthesis", "meaning of ubiquitous", "what is a black hole", "definition of irony"],
        "Narrative": [
            "Movie about astronauts stays in orbit while others go to the surface of a planet where time passes differently",
            "Old movie airplane falls in strange area disgusting food eyes in soup",
            "Girl comes home late from prom lies to mom about being drunk",
            "Successful little person writer lonely in private life falls in love with red-haired woman"
        ],
        "Keyword": ["weather", "amazon", "google maps", "translator"],
        "Temporal": ["when did world war 2 end", "date of last lunar eclipse", "next olympics location", "who won the super bowl 2024"]
    }
    
    queries = []
    for q_list in queries_dict.values():
        queries.extend(q_list)

    print(f"Analyzing {len(queries)} queries across depth (Task-Specific Grid)...")

    # 3. Generate Q-Net
    inputs = tokenizer(queries, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        output = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        q_net = output.representation # NoTorchSequential
    
    # 4. Analyze Layers (Weights vs Biases)
    print("\n--- Layer-wise Adaptation (Weights vs Biases) ---")
    
    layers_data = [] # Store for plotting
    
    for i, layer in enumerate(q_net.layers):
        # Extract Weights & Biases
        weights = None
        biases = None
        layer_type = ""
        
        if hasattr(layer, 'linear'): # NoTorchDenseBlock
            weights = layer.linear.weight
            biases = layer.linear.bias
            layer_type = "DenseBlock"
        elif hasattr(layer, 'weight'): # NoTorchLinear
            weights = layer.weight
            biases = layer.bias
            layer_type = "Linear"
            
        if weights is not None:
            # Stats Function
            def get_rsd(tensor):
                if tensor is None: return 0.0
                mean_abs = torch.mean(torch.abs(tensor))
                param_std = torch.std(tensor, dim=0) 
                mean_param_std = torch.mean(param_std)
                return (mean_param_std / mean_abs).item() * 100

            weight_rsd = get_rsd(weights)
            bias_rsd = get_rsd(biases)
            
            print(f"Layer {i} ({layer_type}): Weight RSD={weight_rsd:.2f}%, Bias RSD={bias_rsd:.2f}%")
            
            layers_data.append({
                "layer": i,
                "weight_rsd": weight_rsd,
                "bias_rsd": bias_rsd
            })

    # Plot Weight vs Bias Adaptation
    l_indices = [d["layer"] for d in layers_data]
    w_rsds = [d["weight_rsd"] for d in layers_data]
    b_rsds = [d["bias_rsd"] for d in layers_data]
    
    # Text Size configuration
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18, 'legend.fontsize': 14})
    
    plt.figure(figsize=(10, 6))
    plt.plot(l_indices, w_rsds, marker='o', label="Weight RSD (Geometric)", linewidth=3, color="#A51E37", markersize=8)
    plt.plot(l_indices, b_rsds, marker='s', label="Bias RSD (Translational)", linewidth=3, linestyle="--", color="#34495E", markersize=8)
    plt.title("Adaptation Mode: Weights vs Biases", pad=20)
    plt.xlabel("Layer Depth", labelpad=10)
    plt.ylabel("Relative Adaptation (RSD %)", labelpad=10)
    plt.legend(frameon=True)
    plt.grid(True, alpha=0.3, linestyle=":")
    plt.tight_layout()
    # Save main plot to images dir for thesis
    output_dir = "msc_thesis/images"
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, "qnet_depth_bias_weight.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path} (with updated style and colors)")

    # Print LaTeX Table
    print("\n--- LaTeX Table for Results Chapter ---")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Relative Standard Deviation (RSD) of Q-Net weights and biases ($N=32$). Adaptation intensity is primarily concentrated in the input layer weights, with a secondary peak in the intermediate bottleneck (Layers 3--4).}")
    print("\\label{tab:qnet_rsd_data}")
    print("\\begin{tabular}{lrr}")
    print("\\hline")
    print("\\textbf{Layer} & \\textbf{Weight RSD (\\%)} & \\textbf{Bias RSD (\\%)} \\\\ \\hline")
    for d in layers_data:
        print(f"Layer {d['layer']} & {d['weight_rsd']:.2f}\\% & {d['bias_rsd']:.2f}\\% \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    # 5. Spectral Analysis of Layer 0 (The Adaptive Lens)
    print("\n--- Spectral Analysis (Layer 0) ---")
    # Layer 0 Weights: (Batch, Out, In) -> (16, 768, 768)
    layer0_weights = q_net.layers[0].linear.weight
    
    # Compute SVD for each query's weight matrix
    # SVs represent the "Principal Components" of the transformation
    # Shape: (Batch, Min(Out, In)) -> (16, 768)
    U, S, V = torch.svd(layer0_weights)
    
    # 1. Effective Rank (Entropy of Singular Values)
    # Are some queries "simpler" (low rank) than others?
    # Normalized SVs
    S_norm = S / torch.sum(S, dim=1, keepdim=True)
    entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-9), dim=1).cpu().numpy()
    
    # 2. Condition Number (Max SV / Min SV)
    # How "distorted" is the space?
    cond_num = (torch.max(S, dim=1).values / torch.min(S, dim=1).values).cpu().numpy()
    
    print("\nSpectral Complexity by Query:")
    for i, q in enumerate(queries):
        print(f"Query: '{q}' | Entropy (Rank): {entropy[i]:.4f} | Cond Num: {cond_num[i]:.4f}")
        
    # Correlation between Query Type and Complexity?
    plt.figure(figsize=(12, 10)) # Increased height for 32 queries
    plt.barh(queries, entropy, color="#3498DB")
    plt.xlabel("Spectral Entropy (Effective Rank)", labelpad=10)
    plt.title("Complexity of Input Projection (Layer 0) by Query", pad=20)
    plt.grid(True, axis='x', alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "qnet_spectral_entropy.png"))
    print(f"Saved {os.path.join(output_dir, 'qnet_spectral_entropy.png')}")
    
    # Plot Eigenspectrum of First vs Last Query
    plt.figure(figsize=(10, 6))
    plt.plot(S[0].cpu().numpy(), label=f"Query: {queries[0]}")
    plt.plot(S[-1].cpu().numpy(), label=f"Query: {queries[-1]}")
    plt.yscale('log')
    plt.title("Singular Value Spectrum (Layer 0)")
    plt.ylabel("Log Magnitude")
    plt.xlabel("Singular Value Index")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "qnet_svd_spectrum.png"))
    print(f"Saved {os.path.join(output_dir, 'qnet_svd_spectrum.png')}")

if __name__ == "__main__":
    analyze_depth()
