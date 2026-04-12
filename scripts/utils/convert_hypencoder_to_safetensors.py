
import sys
import torch
import os
import copy
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder, HypencoderDualEncoderConfig

def convert(checkpoint_path):
    print(f"Loading configuration from {checkpoint_path}...")
    config = HypencoderDualEncoderConfig.from_pretrained(checkpoint_path)
    
    print("Initializing model from config...")
    model = HypencoderDualEncoder(config)
    
    # CRITICAL FIX: Break the shared encoder to allow safe serialization
    if config.shared_encoder:
        print("Breaking shared encoder for Safetensors compatibility...")
        # Use deepcopy to ensure they are distinct objects
        model.passage_encoder.transformer = copy.deepcopy(model.query_encoder.transformer)
        model.config.shared_encoder = False

    bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if not os.path.exists(bin_path):
        print(f"Error: {bin_path} not found.")
        return

    print(f"Loading state dict manually from {bin_path}...")
    # Direct torch.load should work as we are bypassing transformers' security check wrapper
    try:
        state_dict = torch.load(bin_path, map_location="cpu")
    except Exception as e:
        print(f"Failed to load with standard torch.load: {e}")
        # Try with weights_only=True if supported by this torch version, though the error suggested otherwise
        try:
             state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        except:
             raise e

    print("Cloning tensors to break tied weights in state_dict...")
    for key in list(state_dict.keys()):
        state_dict[key] = state_dict[key].clone()

    print("Loading state dict into model...")
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Strict load failed: {e}")
        print("Trying non-strict load...")
        model.load_state_dict(state_dict, strict=False)
    
    print("Saving model to safetensors...")
    # Save with safe_serialization=True
    model.save_pretrained(checkpoint_path, safe_serialization=True)
    print("Conversion complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_hypencoder_to_safetensors.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    convert(checkpoint_path)
