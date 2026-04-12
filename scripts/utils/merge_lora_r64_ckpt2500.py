
import os
import torch
import shutil
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder
from peft import get_peft_model, LoraConfig, TaskType

# Configuration
# Pointing to the SPECIFIC CHECKPOINT (Step 2500)
CHECKPOINT_PATH = "./checkpoints/hypencoder.6_layer_lora_r64/checkpoint-2500"
BASE_MODEL_NAME = "jfkback/hypencoder.6_layer" 
# New output path
OUTPUT_PATH = "./checkpoints/hypencoder.6_layer_lora_r64/merged_checkpoint-2500"

print(f"--- Merging LoRA Weights (r=64, Step 2500) ---")
print(f"Source: {CHECKPOINT_PATH}")
print(f"Base: {BASE_MODEL_NAME}")
print(f"Output: {OUTPUT_PATH}")

# 1. Initialize Base Model
print("Loading base model...")
model = HypencoderDualEncoder.from_pretrained(BASE_MODEL_NAME)

# 2. Configure LoRA
# Values MUST match my_configs/hypencoder.6_layer_lora_r64.yaml
print("Applying LoRA config (r=64, alpha=128)...")
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=64,           # CHANGED from 8
    lora_alpha=128, # CHANGED from 32
    lora_dropout=0.1,
    target_modules=["query", "key", "value", "dense"], 
)

# Apply to query encoder
model.query_encoder.transformer = get_peft_model(model.query_encoder.transformer, peft_config)
# Apply to passage encoder (shared)
model.passage_encoder.transformer = model.query_encoder.transformer

# 3. Load Trained Weights
print(f"Loading state dict from {CHECKPOINT_PATH}...")
bin_path = os.path.join(CHECKPOINT_PATH, "pytorch_model.bin")
if not os.path.exists(bin_path):
    bin_path = os.path.join(CHECKPOINT_PATH, "adapter_model.bin")

print(f"Reading from: {bin_path}")
state_dict = torch.load(bin_path, map_location="cpu")

# Load state dict
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Missing keys: {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")

# 4. Merge and Unload
print("Merging LoRA adapters...")
model.query_encoder.transformer = model.query_encoder.transformer.merge_and_unload()
if model.passage_encoder.transformer is not model.query_encoder.transformer:
     model.passage_encoder.transformer = model.passage_encoder.transformer.merge_and_unload()

# 5. Save
print(f"Saving merged model to {OUTPUT_PATH}...")
model.save_pretrained(OUTPUT_PATH)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.save_pretrained(OUTPUT_PATH)

print(" Done!")
