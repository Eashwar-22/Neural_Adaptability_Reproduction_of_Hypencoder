
import os
import sys
import torch
from safetensors.torch import save_file
from collections import defaultdict

def convert_bin_to_safetensors(checkpoint_dir):
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    out_path = os.path.join(checkpoint_dir, "model.safetensors")

    if not os.path.exists(bin_path):
        print(f"Error: {bin_path} not found.")
        sys.exit(1)
    
    if os.path.exists(out_path):
        print(f"Warning: {out_path} already exists. Overwriting...")

    print(f"Loading {bin_path}...")
    # Load with weights_only=False to bypass strict check if possible, or just standard load
    # The vulnerability error usually comes from higher-level libraries. 
    # Standard torch.load might warn but proceed.
    try:
        state_dict = torch.load(bin_path, map_location="cpu")
    except Exception as e:
        print(f"Failed to load with defaults, trying weights_only=False: {e}")
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)

    print("Processing state dict to handle shared tensors...")
    # Safetensors doesn't like shared tensors (same memory address for different keys).
    # We must clone them to make them distinct.
    
    # Identify shared tensors
    tensor_map = defaultdict(list)
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            ptr = v.data_ptr()
            tensor_map[ptr].append(k)
            
    shared_count = 0
    new_state_dict = {}
    
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            ptr = v.data_ptr()
            if len(tensor_map[ptr]) > 1:
                # This is a shared tensor. Clone it.
                # print(f"Cloning shared tensor: {k}")
                new_state_dict[k] = v.clone()
                shared_count += 1
            else:
                new_state_dict[k] = v
        else:
            new_state_dict[k] = v

    print(f"Cloned {shared_count} shared tensor entries to satisfy Safetensors format.")
    
    print(f"Saving to {out_path}...")
    save_file(new_state_dict, out_path)
    print("Conversion complete.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_bin_to_safetensors.py <checkpoint_dir>")
        sys.exit(1)
    
    convert_bin_to_safetensors(sys.argv[1])
