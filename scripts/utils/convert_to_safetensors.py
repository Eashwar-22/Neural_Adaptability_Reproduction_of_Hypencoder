import sys
import torch
from hypencoder_cb.modeling.hypencoder_colbert import HypencoderDualEncoder

from hypencoder_cb.modeling.hypencoder_colbert import HypencoderDualEncoder, HypencoderDualEncoderConfig
import os

import copy

def convert(checkpoint_path):
    print(f"Loading configuration from {checkpoint_path}...")
    config = HypencoderDualEncoderConfig.from_pretrained(checkpoint_path)
    
    print("Initializing model from config...")
    model = HypencoderDualEncoder(config)
    
    # CRITICAL FIX: Break the shared encoder to allow safe serialization
    # Safetensors struggles with shared weights unless explicitly handled.
    # We force them to be separate modules.
    if config.shared_encoder:
        print("Breaking shared encoder for Safetensors compatibility...")
        # Use deepcopy to ensure they are distinct objects
        model.passage_encoder.transformer = copy.deepcopy(model.query_encoder.transformer)
        # We also need to update the config so it doesn't try to share them on reload (optional but good practice)
        model.config.shared_encoder = False

    bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    print(f"Loading state dict manually from {bin_path}...")
    # Bypass transformers check by using torch.load directly
    state_dict = torch.load(bin_path, map_location="cpu") 
    
    print("Cloning tensors to break tied weights in state_dict...")
    for key in list(state_dict.keys()):
        state_dict[key] = state_dict[key].clone()

    print("Loading state dict into model...")
    model.load_state_dict(state_dict)
    
    print("Saving model to safetensors...")
    # Save with safe_serialization=True
    model.save_pretrained(checkpoint_path, safe_serialization=True)
    print("Conversion complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_safetensors.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    convert(checkpoint_path)
