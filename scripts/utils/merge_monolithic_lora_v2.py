
import os
import sys
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig

# Add project root to path
sys.path.append(os.getcwd())

from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder

def merge_monolithic_lora():
    # Paths
    checkpoint_path = "checkpoints/hypencoder.6_layer_lora_r64_v2/checkpoint-4500"
    output_path = "checkpoints/hypencoder.6_layer_lora_r64_v2/merged_checkpoint-4500"
    
    print(f"Loading monolithic checkpoint from {checkpoint_path}...")
    
    # 1. Load the state dict directly
    state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location="cpu")
    
    # Verify LoRA keys exist
    lora_keys = [k for k in state_dict.keys() if "lora" in k]
    print(f"Found {len(lora_keys)} LoRA keys in state dict.")
    if len(lora_keys) == 0:
        print("ERROR: No LoRA keys found! This is just a base model.")
        return

    # 2. Initialize the Base Model
    # We use from_pretrained which will load the base weights and ignore LoRA keys (with warnings)
    print("Initializing base model from pretrained...")
    base_model = HypencoderDualEncoder.from_pretrained(checkpoint_path)
    
    # 3. Apply LoRA Config
    # We need to match the keys in the state_dict.
    # If state_dict has 'base_model.model.query_encoder...', then PEFT wrapping usually adds 'base_model.model.'
    # Let's inspect the first LoRA key to guess the target modules.
    first_lora_key = lora_keys[0]
    print(f"Sample LoRA key: {first_lora_key}")
    
    # Heuristic to determine target modules from keys
    # Key format: ...layer.0.attention.self.query.lora_A.weight
    targets = set()
    if "query" in first_lora_key: targets.add("query")
    if "key" in first_lora_key: targets.add("key")
    if "value" in first_lora_key: targets.add("value")
    if "dense" in first_lora_key: targets.add("dense") # might coincide with others, check carefully
    # Dense is usually 'output.dense' or 'intermediate.dense'
    
    # Manually specifying common targets 
    target_modules = ["query", "key", "value", "dense"]
    
    peft_config = LoraConfig(
        r=64,
        lora_alpha=64,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=None, # Hypencoder is custom, so None or FEATURE_EXTRACTION
    )
    
    # Wrap model
    model = get_peft_model(base_model, peft_config)
    
    # 4. Load State Dict
    # The state dict has keys like 'base_model.model.query_encoder...' likely?
    # Or just 'query_encoder...'?
    # If the checkpoint is monolithic from a PEFT training, the keys might already be prefixed with 'base_model.model.'
    # Let's inspect a key from the state dict to be sure.
    # Inspect State Dict Keys
    lora_keys = [k for k in state_dict.keys() if "lora" in k]
    print(f"Total LoRA keys in checkpoint: {len(lora_keys)}")
    if len(lora_keys) > 0:
        print(f"Sample Checkpoint Key: {lora_keys[0]}")
    
    # PEFT Model Keys
    model_keys = [k for k in model.state_dict().keys() if "lora" in k]
    print(f"Total PEFT Model keys: {len(model_keys)}")
    if len(model_keys) > 0:
        print(f"Sample PEFT Model Key: {model_keys[0]}")

    # FORCE RENAMING
    # The checkpoint keys usually lack 'base_model.model.' if saved directly from DDP or Trainer sometimes.
    # PEFT wraps it in 'base_model.model.'
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if "lora" in k:
            if not k.startswith("base_model.model."):
                 new_key = "base_model.model." + k
            else:
                 new_key = k
            new_state_dict[new_key] = v
        else:
            # Base weights
            if not k.startswith("base_model.model."):
                # If we use strict=False, we might not need base keys if they are already loaded?
                # But to be safe, let's keep them.
                # PEFT base model is at 'base_model.model.'
                new_key = "base_model.model." + k 
            else:
                new_key = k
            new_state_dict[new_key] = v
            
    print("Renamed keys. Attempting load...")
    load_result = model.load_state_dict(new_state_dict, strict=False)
    print(f"Load Result: {load_result}")
    
    # Verify a LoRA weight is non-zero (if initialized that way) or matches
    # print(f"Checking loaded weight: {model.state_dict()[model_keys[0]][0,0]}")
    
    # 5. Merge and Unload
    print("Merging adapters...")
    model = model.merge_and_unload()
    
    # 6. Save
    print(f"Saving merged model to {output_path}...")
    model.save_pretrained(output_path)
    # Save tokenizer too
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.save_pretrained(output_path)
    
    print("Done!")

if __name__ == "__main__":
    merge_monolithic_lora()
