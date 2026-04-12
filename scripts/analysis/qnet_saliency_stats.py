
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder
import random

def load_random_samples(filepath, num_samples=100):
    samples = []
    print(f"Loading {num_samples} random samples from {filepath} (Reservoir Sampling)...")
    
    # Reservoir Sampling to avoid reading all lines to memory
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                print(f"DEBUG First Line: {line[:500]}...")
            
            if i < num_samples:
                try:
                    item = json.loads(line)
                    # Debug print for first item
                    if i == 0:
                        print(f"DEBUG Keys: {item.keys()}")
                        if 'items' in item:
                            print(f"DEBUG Items: {len(item.get('items', []))}")
                            if len(item['items']) > 0:
                                print(f"DEBUG Item 0 Keys: {item['items'][0].keys()}")
                                
                    if 'pos' in item: # Maybe it's 'pos' not 'items'
                         doc = item['pos'][0] if isinstance(item['pos'], list) else item['pos']
                         samples.append({"query": item['query'], "items": [{"content": doc}]})
                    elif 'items' in item and len(item['items']) > 0:
                        samples.append(item)
                except Exception as e:
                    print(f"Error parsing line {i}: {e}")
                    continue
            else:
                j = random.randint(0, i)
                if j < num_samples:
                    try:
                        item = json.loads(line)
                        if 'items' in item and len(item['items']) > 0:
                            samples[j] = item
                    except:
                        continue
            
            # Stop early to save time if file is huge (just take from first 100k)
            if i > 50000: 
                break
    
    final_samples = []
    for item in samples:
        # Fix: Query is a dict or string
        q_obj = item.get('query', '')
        if isinstance(q_obj, dict):
            query = q_obj.get('content', '')
        else:
            query = str(q_obj)
            
        # Fix: Doc is in items[0]
        if 'items' in item and len(item['items']) > 0:
            doc = item['items'][0].get('content', '')
        else:
            doc = ""
            
        if query and doc:
            final_samples.append((query, doc))
            
    return final_samples

def main():
    # Settings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "/home/eickhoff/esx510/hypencoder-paper/checkpoints/hypencoder.6_layer_full_real_opt"
    data_path = "data/triples.train.jsonl"
    output_dir = "docs/analysis/saliency_stats"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    samples = load_random_samples(data_path, num_samples=100)
    print(f"Loaded {len(samples)} valid samples.")
    
    # 2. Load Model
    print(f"Loading model values...")
    model_dual = HypencoderDualEncoder.from_pretrained(checkpoint_path, use_safetensors=True).to(device)
    model_dual.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    distances = []
    diffs = []
    
    print("Computing saliency alignment...")
    for i, (query_text, doc_text) in enumerate(samples):
        if i % 10 == 0: print(f"Processing {i}/{len(samples)}...")
        
        try:
            # A. Encode Query
            q_inputs = tokenizer(query_text, return_tensors="pt").to(device)
            with torch.no_grad():
                q_out = model_dual.query_encoder(q_inputs.input_ids, q_inputs.attention_mask)
                q_net = q_out.representation 
                q_trans_out = model_dual.query_encoder.transformer(q_inputs.input_ids, q_inputs.attention_mask)
                q_emb = q_trans_out.last_hidden_state[:, 0]

            # B. Encode Document (With Gradients)
            # Truncate to avoid OOM in loop
            d_inputs = tokenizer(doc_text, return_tensors="pt", truncation=True, max_length=128).to(device)
            emb_layer = model_dual.passage_encoder.transformer.embeddings
            d_embeds = emb_layer(d_inputs.input_ids)
            d_embeds.retain_grad()
            d_embeds.requires_grad_(True)
            
            transformer_out = model_dual.passage_encoder.transformer(
                inputs_embeds=d_embeds,
                attention_mask=d_inputs.attention_mask
            )
            doc_repr = transformer_out.last_hidden_state[:, 0]
            
            def get_grads(score_tensor, inputs):
                if inputs.grad is not None: inputs.grad.zero_()
                score_tensor.backward(retain_graph=True)
                return inputs.grad.clone()

            # Hypencoder
            doc_repr_unsqueezed = doc_repr.unsqueeze(1) 
            score_hyp = q_net(doc_repr_unsqueezed)
            grad_hyp = get_grads(score_hyp, d_embeds)
            
            # Baseline
            d_embeds.grad.zero_()
            score_base = torch.sum(doc_repr * q_emb)
            grad_base = get_grads(score_base, d_embeds)
            
            # Saliency Calculation
            saliency_hyp = torch.norm(grad_hyp, dim=2).squeeze().cpu().numpy()
            saliency_base = torch.norm(grad_base, dim=2).squeeze().cpu().numpy()
            
            # Cosine Similarity
            # Flatten in case of batch glitches, though we run bs=1
            vec_hyp = saliency_hyp.flatten()
            vec_base = saliency_base.flatten()
            
            if np.linalg.norm(vec_hyp) == 0 or np.linalg.norm(vec_base) == 0:
                continue
                
            cos_sim = np.dot(vec_hyp, vec_base) / (np.linalg.norm(vec_hyp) * np.linalg.norm(vec_base))
            distances.append(cos_sim)
            
            # Difference Metric (Mean Absolute Difference on Normalized Vectors)
            # 1. Normalize to [0,1] to match the visual bar plots
            def normalize(arr):
                if arr.max() - arr.min() == 0: return arr
                return (arr - arr.min()) / (arr.max() - arr.min())
                
            norm_hyp = normalize(vec_hyp)
            norm_base = normalize(vec_base)
            
            # 2. Calculate Mean Absolute Difference for this sample
            # represents "Average change in attention per token"
            mad = np.mean(np.abs(norm_hyp - norm_base))
            diffs.append(mad)
            
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue

    # 3. Statistics & Visualization (Cosine)
    mean_sim = np.mean(distances)
    std_sim = np.std(distances)
    print(f"\nFinal Results (N={len(distances)}):")
    print(f"Mean Cosine Similarity: {mean_sim:.4f}")
    print(f"Std Deviation: {std_sim:.4f}")
    
    # Statistics (Difference)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    print(f"Mean Absolute Diff (Norm): {mean_diff:.4f}")
    print(f"Std Deviation (Diff): {std_diff:.4f}")
    
    # Save Cosine Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(distances, bins=20, kde=True, color='purple')
    plt.axvline(mean_sim, color='k', linestyle='--', linewidth=2, label=f'Mean: {mean_sim:.3f}')
    plt.title(f"Distribution of Saliency Alignment (N={len(distances)})\nHypencoder vs Bi-Encoder")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(alpha=0.3)
    
    filename = f"{output_dir}/cosine_histogram.png"
    plt.savefig(filename)
    print(f"Saved histogram: {filename}")
    
    # Save Difference Histogram
    plt.clf()
    plt.figure(figsize=(10, 6))
    sns.histplot(diffs, bins=20, kde=True, color='orange')
    plt.axvline(mean_diff, color='k', linestyle='--', linewidth=2, label=f'Mean: {mean_diff:.3f}')
    plt.title(f"Distribution of Saliency Differences (N={len(diffs)})\nMean Absolute Difference (Normalized)")
    plt.xlabel("Mean Abs Difference (0.0 to 1.0)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(alpha=0.3)
    
    filename_diff = f"{output_dir}/diff_histogram.png"
    plt.savefig(filename_diff)
    print(f"Saved diff histogram: {filename_diff}")
    
    # Save raw data
    with open(f"{output_dir}/stats_raw.json", "w") as f:
        json.dump({
            "cosine_sim": [float(x) for x in distances],
            "abs_diff": [float(x) for x in diffs]
        }, f)

if __name__ == "__main__":
    main()
