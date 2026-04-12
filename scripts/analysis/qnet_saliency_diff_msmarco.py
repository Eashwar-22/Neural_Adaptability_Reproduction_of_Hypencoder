
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder

def get_cases():
    return [
        (
            "meaning of name lysander",
            "Latin Meaning: The name Lysander is a Latin baby name. In Latin the meaning of the name Lysander is: Liberator. Greek Meaning: The name Lysander is a Greek baby name. In Greek the meaning of the name Lysander is: Liberator.Lysander is one of the main characters in Shakespeare's A Midsummer Night's Dream. Shakespearean Meaning: The name Lysander is a Shakespearean baby name. In Shakespearean the meaning of the name Lysander is: A Midsummer Night's Dream' In love with Hermia.n Greek the meaning of the name Lysander is: Liberator. Lysander is one of the main characters in Shakespeare's A Midsummer Night's Dream. Shakespearean Meaning: The name Lysander is a Shakespearean baby name."
        ),
        (
            "how about syracuse university?",
            "Syracuse University, commonly referred to as Syracuse, 'Cuse, or SU, is a private research university in Syracuse, New York, United States. The institution's roots can be traced to the Genesee Wesleyan Seminary (later becoming Genesee College), founded in 1831 by the Methodist Episcopal Church in Lima, New York. After several years of debate over relocating the college to Syracuse, the university was established in 1870, independent of the college. Since 1920, the university has identified itself as nonsectarian, although it maintains a relationship with The United Methodist Church."
        ),
        (
            "prefix definition of im-",
            "Prefix im- is for opposite or asserting. It should be noted that the prefix im- is really just the prefix in- with the consonant adjusted because of the consonant that follows. As for passive and impassive, it's a quirk of the history of the meaning that they've ended up with similar meanings."
        ),
        (
            "what county is richland mi",
            "This article is about the village in Kalamazoo County. For the five townships named Richland, see Richland Township, Michigan (disambiguation). Richland is a village in Kalamazoo County in the U.S. state of Michigan. The population was 791 at the 2010 census. The village is within Richland Township, about nine miles northeast of Kalamazoo."
        ),
        (
            "psychotherapy fees average",
            "the fee for services charged for clinical professions can vary due to a number of factors for example by geographic location type of therapist lcsw phd md and type of therapy generally speaking individual psychotherapy sessions range from about $ 90 and up per session in the dc metropolitan area"
        )
    ]

def main():
    # Settings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "./checkpoints/hypencoder.6_layer_full_real_opt"
    output_dir = "docs/analysis/saliency_cosine"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    samples = get_cases()
    print(f"Analyzing {len(samples)} specific cases...")
    
    # 2. Load Model
    print(f"Loading model from {checkpoint_path}...")
    model_dual = HypencoderDualEncoder.from_pretrained(checkpoint_path, use_safetensors=True).to(device)
    model_dual.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # 3. Analyze Each Sample
    summary_stats = []
    
    for idx, (query_text, doc_text) in enumerate(samples):
        print(f"\nProcessing Sample {idx+1}: '{query_text}'")
        
        # A. Encode Query
        q_inputs = tokenizer(query_text, return_tensors="pt").to(device)
        with torch.no_grad():
            q_out = model_dual.query_encoder(q_inputs.input_ids, q_inputs.attention_mask)
            q_net = q_out.representation 
            q_trans_out = model_dual.query_encoder.transformer(q_inputs.input_ids, q_inputs.attention_mask)
            q_emb = q_trans_out.last_hidden_state[:, 0]

        # B. Encode Document (With Gradients)
        d_inputs = tokenizer(doc_text, return_tensors="pt", truncation=True, max_length=128).to(device)
        emb_layer = model_dual.passage_encoder.transformer.embeddings
        d_embeds = emb_layer(d_inputs.input_ids)
        d_embeds.retain_grad()
        d_embeds.requires_grad_(True)
        
        transformer_out = model_dual.passage_encoder.transformer(
            inputs_embeds=d_embeds,
            attention_mask=d_inputs.attention_mask
        )
        last_hidden_state = transformer_out.last_hidden_state
        doc_repr = last_hidden_state[:, 0]
        
        # Helper for clean Backward
        def get_grads(score_tensor, inputs):
            if inputs.grad is not None: inputs.grad.zero_()
            score_tensor.backward(retain_graph=True)
            return inputs.grad.clone()

        # Method 1: Hypencoder
        doc_repr_unsqueezed = doc_repr.unsqueeze(1) 
        score_hyp = q_net(doc_repr_unsqueezed)
        grad_hyp = get_grads(score_hyp, d_embeds)
        
        # Method 2: Baseline (Dot Product)
        d_embeds.grad.zero_()
        score_base = torch.sum(doc_repr * q_emb)
        grad_base = get_grads(score_base, d_embeds)
        
        # Calculate Norms (Saliency Vectors)
        saliency_hyp = torch.norm(grad_hyp, dim=2).squeeze().cpu().numpy()
        saliency_base = torch.norm(grad_base, dim=2).squeeze().cpu().numpy()
        
        # 4. Calculate Cosine Similarity between the two Saliency Vectors
        # This measures how similar the *patterns* of attention are aligned.
        
        # Normalize vectors for calculation
        vec_hyp = saliency_hyp
        vec_base = saliency_base
        
        cos_sim = np.dot(vec_hyp, vec_base) / (np.linalg.norm(vec_hyp) * np.linalg.norm(vec_base))
        print(f"  Cosine Similarity (Pattern Alignment): {cos_sim:.4f}")
        
        summary_stats.append({
            "query": query_text,
            "cosine_sim": float(cos_sim)
        })

        # 5. Visualization: Dual Line Plot
        # To visually verify alignment, we plot BOTH lines normalized to [0,1].
        
        def normalize(arr):
            return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
            
        norm_hyp = normalize(saliency_hyp)
        norm_base = normalize(saliency_base)
        
        tokens = tokenizer.convert_ids_to_tokens(d_inputs.input_ids[0])
        x = range(len(tokens))
        
        plt.figure(figsize=(15, 6))
        plt.plot(x, norm_hyp, label=f"Hypencoder", color="green", linewidth=2, marker='o', markersize=4, alpha=0.8)
        plt.plot(x, norm_base, label=f"Baseline (Bi-Encoder)", color="red", linewidth=2, linestyle="--", marker='x', markersize=4, alpha=0.6)
        
        plt.xticks(x, tokens, rotation=90, fontsize=8)
        plt.title(f"Saliency Alignment (Cos Sim: {cos_sim:.4f})\nQuery: {query_text}")
        plt.ylabel("Normalized Saliency")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        filename = f"{output_dir}/cosine_sample_{idx+1}.png"
        plt.savefig(filename)
        print(f"Saved plot: {filename}")
        
    # Save Summary
    with open(f"{output_dir}/cosine_stats.json", "w") as f:
        json.dump(summary_stats, f, indent=2)

if __name__ == "__main__":
    main()
