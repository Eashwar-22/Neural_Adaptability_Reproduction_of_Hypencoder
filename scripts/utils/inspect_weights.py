
import torch
from transformers import AutoModel
import os
import sys

# Add project root to path so we can import hypencoder_cb
sys.path.append(os.getcwd())

from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder

def check_model(model_path):
    print(f"Checking model at: {model_path}")
    
    try:
        # Load the custom model (merged)
        # We perform a strict load to see if keys are missing
        model = HypencoderDualEncoder.from_pretrained(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Check against random initialization (if weights are just random)
    # or against base BERT
    
    # Let's inspect the query_encoder weights
    print("\nVerifying weights...")
    
    # Load base BERT to compare
    base_bert = AutoModel.from_pretrained("google-bert/bert-base-uncased")
    
    # In HypencoderDualEncoder, the BERT part is in model.query_encoder.transformer
    custom_bert = model.query_encoder.transformer
    
    diff_count = 0
    total_count = 0
    
    for key in base_bert.state_dict().keys():
        if key in custom_bert.state_dict():
            total_count += 1
            base_param = base_bert.state_dict()[key]
            custom_param = custom_bert.state_dict()[key]
            
            if not torch.equal(base_param, custom_param):
                diff_count += 1
                # print(f"  Diff: {key}")
                
    print(f"Total BERT keys compared: {total_count}")
    print(f"Different keys (modified from BERT): {diff_count}")
    
    if diff_count == 0:
        print("CRITICAL WARNING: The model contains PURE BERT COMPATIBLE WEIGHTS!")
        print("This means the LoRA merge did not update the base weights, or they were lost on save.")
    else:
        print(f"SUCCESS: The model differs from BERT in {diff_count} parameters.")

    # Also check if the Hyper-Network heads are random or learned
    # (Checking against 0 is a quick heuristic if they were initialized to 0)
    # But checking against a re-initialized model is better.
    
    # Check if hyper_base_matrices are all zeros (some orig init is zero?)
    # matrix = model.query_encoder.hyper_base_matrices[0]
    # print(f"Hyper Matrix 0 Mean: {matrix.mean().item()}")

if __name__ == "__main__":
    check_model("checkpoints/hypencoder.6_layer_lora_r64_v2/merged_checkpoint-4500")
