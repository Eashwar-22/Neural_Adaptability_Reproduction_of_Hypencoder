
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from transformers import AutoTokenizer
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder

# Style Configuration
THEME_COLOR = "#A51E37"  # Thesis "rot"
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 12,
    'figure.dpi': 300
})

def generate_plots():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "./checkpoints/hypencoder.6_layer_full_real_opt"
    output_dir = "./msc_thesis/images/new_plots"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model and tokenizer...")
    model_dual = HypencoderDualEncoder.from_pretrained(checkpoint_path, use_safetensors=True).to(device)
    model = model_dual.query_encoder
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model.eval()

    # 1. Categories and Queries
    categories = {
        "Medical/Broad": ["symptoms of covid", "common cold treatments", "how heart works"],
        "Technical": ["docker run command", "how to sort a list in python", "git commit message"],
        "Entity/Factoid": ["when was barack obama born", "who wrote the great gatsby", "capital of france"]
    }
    
    flat_queries = []
    query_to_cat = []
    for cat, q_list in categories.items():
        flat_queries.extend(q_list)
        query_to_cat.extend([cat] * len(q_list))

    print(f"Processing {len(flat_queries)} queries...")
    inputs = tokenizer(flat_queries, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        output = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        q_net = output.representation 

    # --- PLOT 1: Weight vs Bias RSD ---
    print("Generating Plot 1: Weight vs Bias RSD...")
    layers_data = []
    for i, layer in enumerate(q_net.layers):
        weights = None
        biases = None
        if hasattr(layer, 'linear'):
            weights = layer.linear.weight
            biases = layer.linear.bias
        elif hasattr(layer, 'weight'):
            weights = layer.weight
            biases = layer.bias

        if weights is not None:
            def get_rsd(tensor):
                if tensor is None: return 0.0
                mean_abs = torch.mean(torch.abs(tensor))
                param_std = torch.std(tensor, dim=0) 
                mean_param_std = torch.mean(param_std)
                return (mean_param_std / mean_abs).item() * 100

            layers_data.append({
                "layer": i,
                "weight_rsd": get_rsd(weights),
                "bias_rsd": get_rsd(biases)
            })

    l_indices = [d["layer"] for d in layers_data]
    w_rsds = [d["weight_rsd"] for d in layers_data]
    b_rsds = [d["bias_rsd"] for d in layers_data]

    plt.figure(figsize=(8, 5))
    plt.plot(l_indices, w_rsds, marker='o', color=THEME_COLOR, label="Weight RSD (Geometric)", linewidth=2)
    plt.plot(l_indices, b_rsds, marker='s', color='#2F4F4F', label="Bias RSD (Translational)", linewidth=2, linestyle="--")
    plt.xlabel("Layer Depth")
    plt.ylabel("Relative Adaptation (RSD %)")
    plt.title("Adaptation Mode: Weights vs Biases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "qnet_weight_vs_bias_rsd.png"))
    plt.close()

    # --- PLOT 2: Singular Value Spectrum ---
    print("Generating Plot 2: Singular Value Spectrum...")
    q_medical_idx = flat_queries.index("symptoms of covid")
    q_entity_idx = flat_queries.index("who wrote the great gatsby")
    
    layer0_weights = q_net.layers[0].linear.weight
    _, S_medical, _ = torch.svd(layer0_weights[q_medical_idx])
    _, S_entity, _ = torch.svd(layer0_weights[q_entity_idx])

    plt.figure(figsize=(8, 5))
    plt.plot(S_medical.cpu().numpy(), color='#2F4F4F', label="Query: Medical (Broad)", linewidth=2)
    plt.plot(S_entity.cpu().numpy(), color=THEME_COLOR, label="Query: Entity (Specific)", linewidth=2)
    plt.yscale('log')
    plt.xlabel("Singular Value Index")
    plt.ylabel("Magnitude (Log Scale)")
    plt.title("Singular Value Spectrum (Layer 0 Rank)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "qnet_svd_spectrum_log.png"))
    plt.close()

    # --- PLOT 3: Spectral Entropy by Category ---
    print("Generating Plot 3: Spectral Entropy by Category...")
    _, S_batch, _ = torch.svd(layer0_weights)
    S_norm = S_batch / torch.sum(S_batch, dim=1, keepdim=True)
    entropies = -torch.sum(S_norm * torch.log(S_norm + 1e-9), dim=1).cpu().numpy()
    
    avg_entropies = []
    cat_labels = list(categories.keys())
    for cat in cat_labels:
        indices = [i for i, c in enumerate(query_to_cat) if c == cat]
        avg_entropies.append(np.mean(entropies[indices]))

    plt.figure(figsize=(8, 5))
    bars = plt.bar(cat_labels, avg_entropies, color=[THEME_COLOR, '#2F4F4F', '#708090'])
    plt.ylabel("Average Spectral Entropy")
    plt.title("Complexity of Retrieval by Category")
    plt.ylim(5.5, 6.5)  # Zoom in on differences
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "qnet_entropy_by_category.png"))
    plt.close()

    print(f"All plots saved to {output_dir}")

if __name__ == "__main__":
    generate_plots()
