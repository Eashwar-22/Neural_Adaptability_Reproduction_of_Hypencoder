
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import html
from transformers import AutoTokenizer
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder

def analyze_saliency():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Model
    checkpoint_path = "/home/eickhoff/esx510/hypencoder-paper/checkpoints/hypencoder.6_layer_full_real_opt"
    print(f"Loading model from {checkpoint_path}...")
    
    model_dual = HypencoderDualEncoder.from_pretrained(checkpoint_path, use_safetensors=True).to(device)
    model_dual.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # 2. Define Cases
    categories = {
        "Entity": [
            ("who wrote the great gatsby", "The Great Gatsby is a 1925 novel by American writer F. Scott Fitzgerald. It follows a cast of characters living in the fictional towns of West Egg and East Egg."),
            ("height of mount everest", "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. Its elevation is 8,848.86 m."),
            ("founder of microsoft", "Microsoft Corporation is an American multinational technology corporation which produces computer software. It was founded by Bill Gates and Paul Allen.")
        ],
        "Technical": [
            ("how to install docker ubuntu", "To install Docker Engine on Ubuntu, ensure you meet the prerequisites. Then run: sudo apt-get update && sudo apt-get install docker-ce docker-ce-cli."),
            ("git commit message convention", "The Conventional Commits specification is a lightweight convention on top of commit messages. It provides an easy set of rules for creating an explicit commit history."),
            ("python sort list descending", "To sort a list in descending order in Python, you can use the sort() method with reverse=True. Example: my_list.sort(reverse=True).")
        ],
        "Medical": [
             ("symptoms of covid 19", "People with COVID-19 have had a wide range of symptoms reported – ranging from mild symptoms to severe illness. Symptoms may include fever, cough, and shortness of breath."),
             ("side effects of ibuprofen", "Common side effects of ibuprofen include nausea, vomiting, diarrhea, and indigestion. Serious side effects include stomach ulcers and bleeding."),
             ("benefits of meditation", "Meditation can produce a deep state of relaxation and a tranquil mind. Benefits include reduced stress, controlled anxiety, and improved emotional health.")
        ]
    }

    print(f"Analyzing {len(categories)} categories with {sum(len(v) for v in categories.values())} total cases...")

    results = {}

    for cat_name, cases in categories.items():
        print(f"\n--- Category: {cat_name} ---")
        sims = []
        
        for i, (query_text, doc_text) in enumerate(cases):
            # A. Encode Query -> Q-Net
            q_inputs = tokenizer(query_text, return_tensors="pt").to(device)
            with torch.no_grad():
                q_out = model_dual.query_encoder(q_inputs.input_ids, q_inputs.attention_mask)
                q_net = q_out.representation 
                
                q_trans_out = model_dual.query_encoder.transformer(q_inputs.input_ids, q_inputs.attention_mask)
                q_emb = q_trans_out.last_hidden_state[:, 0]

            # B. Encode Document (With Gradients)
            d_inputs = tokenizer(doc_text, return_tensors="pt").to(device)
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
            
            # Method 2: Baseline
            d_embeds.grad.zero_()
            score_base = torch.sum(doc_repr * q_emb)
            grad_base = get_grads(score_base, d_embeds)
            
            # Process & Compare
            saliency_hyp = torch.norm(grad_hyp, dim=2).squeeze().cpu().numpy()
            saliency_base = torch.norm(grad_base, dim=2).squeeze().cpu().numpy()
            
            sim = np.dot(saliency_hyp, saliency_base) / (np.linalg.norm(saliency_hyp) * np.linalg.norm(saliency_base))
            sims.append(sim)
            print(f"  Query: '{query_text}' | Sim: {sim:.4f}")
            
            # Save First Case as Representative Plot
            if i == 0:
                saliency_hyp = (saliency_hyp - saliency_hyp.min()) / (saliency_hyp.max() - saliency_hyp.min() + 1e-9)
                saliency_base = (saliency_base - saliency_base.min()) / (saliency_base.max() - saliency_base.min() + 1e-9)
                tokens = tokenizer.convert_ids_to_tokens(d_inputs.input_ids[0])
                
                plt.figure(figsize=(12, 4))
                x = range(len(tokens))
                plt.plot(x, saliency_hyp, label="Hypencoder", marker="o")
                plt.plot(x, saliency_base, label="Baseline (Dot)", marker="x", linestyle="--")
                plt.xticks(x, tokens, rotation=90)
                plt.title(f"Saliency Profile ({cat_name}): {query_text}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"saliency_{cat_name}.png")
                print(f"  Saved plot: saliency_{cat_name}.png")

        results[cat_name] = {
            "mean": np.mean(sims),
            "std": np.std(sims)
        }

    print("\n--- Final Aggregated Results ---")
    print(f"{'Category':<15} | {'Mean Sim':<10} | {'Std Dev':<10}")
    print("-" * 40)
    for cat, stats in results.items():
        print(f"{cat:<15} | {stats['mean']:.4f}     | {stats['std']:.4f}")

if __name__ == "__main__":
    analyze_saliency()
