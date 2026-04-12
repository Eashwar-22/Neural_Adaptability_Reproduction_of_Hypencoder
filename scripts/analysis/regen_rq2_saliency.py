import sys
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

# Add the project root to sys.path to import hypencoder_cb
project_root = "/home/eickhoff/esx510/hypencoder-paper"
sys.path.append(project_root)

from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder

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
    
    # 1. Hypencoder Saliency
    doc_repr_unsqueezed = doc_repr.unsqueeze(1) 
    score_hyp = q_repr(doc_repr_unsqueezed)
    grad_hyp = get_grads(score_hyp, d_embeds)
    saliency_hyp = torch.norm(grad_hyp, dim=2).squeeze().cpu().numpy()
    saliency_hyp = (saliency_hyp - saliency_hyp.min()) / (saliency_hyp.max() - saliency_hyp.min() + 1e-9)
    
    # 2. Control/Baseline Saliency (Dot Product)
    score_ctrl = torch.sum(doc_repr * q_emb)
    grad_ctrl = get_grads(score_ctrl, d_embeds)
    saliency_ctrl = torch.norm(grad_ctrl, dim=2).squeeze().cpu().numpy()
    saliency_ctrl = (saliency_ctrl - saliency_ctrl.min()) / (saliency_ctrl.max() - saliency_ctrl.min() + 1e-9)
    
    return saliency_hyp, saliency_ctrl, tokenizer.convert_ids_to_tokens(d_inputs.input_ids[0])

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpoint for RQ2 (Official MiniLM Distilled)
    ckpt = "/home/eickhoff/esx510/hypencoder-paper/checkpoints/hypencoder.6_layer_full_real_opt"
    
    cases = [
        ("Medical", "symptoms of covid 19", "People with COVID-19 have had a wide range of symptoms reported – ranging from mild symptoms to severe illness. Symptoms may include fever, cough, and shortness of breath."),
        ("Technical", "how to install docker ubuntu", "To install Docker Engine on Ubuntu, ensure you meet the prerequisites. Then run: sudo apt-get update && sudo apt-get install docker-ce docker-ce-cli."),
        ("Entity", "who wrote the great gatsby", "The Great Gatsby is a 1925 novel by American writer F. Scott Fitzgerald. It follows a cast of characters living in the fictional towns of West Egg and East Egg.")
    ]
    
    output_dir = "/home/eickhoff/esx510/hypencoder-paper/msc_thesis/images"
    model = HypencoderDualEncoder.from_pretrained(ckpt).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    
    for i, (case_name, q_text, d_text) in enumerate(cases):
        print(f"Generating RQ2 Saliency for: {case_name}")
        saliency_hyp, saliency_ctrl, tokens = compute_saliency(model, tokenizer, q_text, d_text, device)
        
        plt.figure(figsize=(10, 4))
        x = range(len(tokens))
        plt.plot(x, saliency_hyp, label="Hypencoder", marker="o", color="C1")
        plt.plot(x, saliency_ctrl, label="Control BE", marker="x", linestyle="--", color="C0")
        
        plt.xticks(x, tokens, rotation=90, fontsize=8)
        plt.title(f"Saliency Profile ({case_name}): {q_text}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save as original filename to overwrite with new style
        save_path = f"{output_dir}/saliency_plot_case_{i}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()
