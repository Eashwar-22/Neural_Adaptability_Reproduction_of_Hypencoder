import sys
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

# Add the project root to sys.path to import hypencoder_cb
project_root = "."
sys.path.append(project_root)

from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder, TextDualEncoder

def get_grads(score_tensor, inputs):
    if inputs.grad is not None: inputs.grad.zero_()
    score_tensor.backward(retain_graph=True)
    return inputs.grad.clone()

def compute_saliency(model, tokenizer, query_text, doc_text, device):
    # A. Encode Query
    q_inputs = tokenizer(query_text, return_tensors="pt").to(device)
    with torch.no_grad():
        q_out = model.query_encoder(q_inputs.input_ids, q_inputs.attention_mask)
        q_repr = q_out.representation 
        
        # If it's a Hypencoder, q_repr is the Q-Net. If it's a Bi-Encoder, it's just the embedding.
        # But we need q_emb (the BERT CLS) for the dot product if it's a Bi-Encoder.
        if isinstance(model, TextDualEncoder):
            q_emb = q_repr
        else:
            # For HypencoderDualEncoder, we need the CLS embedding for the internal BERT logic
            q_trans_out = model.query_encoder.transformer(q_inputs.input_ids, q_inputs.attention_mask)
            q_emb = q_trans_out.last_hidden_state[:, 0]

    # B. Encode Document (With Gradients)
    d_inputs = tokenizer(doc_text, return_tensors="pt").to(device)
    emb_layer = model.passage_encoder.transformer.embeddings
    d_embeds = emb_layer(d_inputs.input_ids)
    d_embeds.retain_grad()
    d_embeds.requires_grad_(True)
    
    transformer_out = model.passage_encoder.transformer(
        inputs_embeds=d_embeds,
        attention_mask=d_inputs.attention_mask
    )
    doc_repr = transformer_out.last_hidden_state[:, 0]
    
    if isinstance(model, HypencoderDualEncoder):
        # Hypencoder logic
        doc_repr_unsqueezed = doc_repr.unsqueeze(1) 
        score = q_repr(doc_repr_unsqueezed)
    else:
        # Bi-Encoder logic (Dot Product)
        score = torch.sum(doc_repr * q_emb)
        
    grad = get_grads(score, d_embeds)
    saliency = torch.norm(grad, dim=2).squeeze().cpu().numpy()
    
    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-9)
    return saliency, tokenizer.convert_ids_to_tokens(d_inputs.input_ids[0])

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cases = [
        ("Medical", "symptoms of covid 19", "People with COVID-19 have had a wide range of symptoms reported – ranging from mild symptoms to severe illness. Symptoms may include fever, cough, and shortness of breath."),
        ("Technical", "how to install docker ubuntu", "To install Docker Engine on Ubuntu, ensure you meet the prerequisites. Then run: sudo apt-get update && sudo apt-get install docker-ce docker-ce-cli."),
        ("Entity", "who wrote the great gatsby", "The Great Gatsby is a 1925 novel by American writer F. Scott Fitzgerald. It follows a cast of characters living in the fictional towns of West Egg and East Egg.")
    ]
    
    checkpoints = {
        "BM25": {
            "Ctrl": "./checkpoints/control_be_bm25",
            "Hyp": "./checkpoints/distill_bm25"
        },
        "BGE": {
            "Ctrl": "./checkpoints/control_be_teacher_sota",
            "Hyp": "./checkpoints/hypencoder_teacher_sota"
        },
        "MXBAI": {
            "Ctrl": "./checkpoints/control_be_teacher_mxbai",
            "Hyp": "./checkpoints/hypencoder_teacher_mxbai"
        }
    }
    
    output_dir = "./msc_thesis/images/saliency_rq3"
    os.makedirs(output_dir, exist_ok=True)
    
    for case_name, q_text, d_text in cases:
        print(f"Processing Case: {case_name}")
        # One plot per case with 3 subplots (one per teacher)
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        for r, teacher in enumerate(["BM25", "BGE", "MXBAI"]):
            ax = axes[r]
            tokens = None
            
            for c, model_type in enumerate(["Ctrl", "Hyp"]):
                ckpt = checkpoints[teacher][model_type]
                print(f"  Loading {teacher} {model_type} from {ckpt}...")
                
                # Determine Model Class using JSON Config
                import json
                with open(os.path.join(ckpt, "config.json"), "r") as f:
                    config_dict = json.load(f)
                
                if "HypencoderDualEncoder" in config_dict.get("architectures", []):
                    model = HypencoderDualEncoder.from_pretrained(ckpt).to(device)
                else:
                    model = TextDualEncoder.from_pretrained(ckpt).to(device)
                
                model.eval()
                tokenizer = AutoTokenizer.from_pretrained(ckpt)
                
                saliency, current_tokens = compute_saliency(model, tokenizer, q_text, d_text, device)
                if tokens is None: tokens = current_tokens
                
                x = range(len(tokens))
                if model_type == "Hyp":
                    ax.plot(x, saliency, label="Hypencoder", marker="o", color="C1")
                else:
                    ax.plot(x, saliency, label="Control BE", marker="x", linestyle="--", color="C0")
                
                # Cleanup to save memory
                del model
                torch.cuda.empty_cache()
            
            ax.set_title(f"Teacher: {teacher}", fontsize=12)
            ax.legend(loc="upper right")
            ax.grid(alpha=0.3)
            
            if r == 2: # Last row
                ax.set_xticks(x)
                ax.set_xticklabels(tokens, rotation=90, fontsize=9)
        
        plt.suptitle(f"Saliency Evolution ({case_name}): {q_text}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = f"{output_dir}/saliency_comp_{case_name.lower()}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()
