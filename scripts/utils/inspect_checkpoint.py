import torch
import sys

ckpt_path = "./checkpoints/hypencoder.6_layer_lora/checkpoint-4000/pytorch_model.bin"
print(f"Loading {ckpt_path}...")
state_dict = torch.load(ckpt_path, map_location="cpu")

lora_keys = [k for k in state_dict.keys() if "lora" in k]
print(f"Total keys: {len(state_dict)}")
print(f"LoRA keys found: {len(lora_keys)}")

if len(lora_keys) > 0:
    print("Example LoRA keys:")
    for k in lora_keys[:5]:
        print(k)
else:
    print("NO LoRA keys found!")
    # Print some random keys to see what's in there
    print("Example keys:")
    for k in list(state_dict.keys())[:5]:
        print(k)
