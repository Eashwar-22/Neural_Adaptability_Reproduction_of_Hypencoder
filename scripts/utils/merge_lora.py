
import os
import torch
import shutil
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder
from peft import get_peft_model, LoraConfig, TaskType

# Configuration
CHECKPOINT_PATH = "./checkpoints/hypencoder.6_layer_lora/checkpoint-4000"
BASE_MODEL_NAME = "jfkback/hypencoder.6_layer" # Using the 6-layer base as per config
OUTPUT_PATH = "./checkpoints/hypencoder.6_layer_lora/merged_checkpoint-4000"

print(f"--- Merging LoRA Weights ---")
print(f"Source: {CHECKPOINT_PATH}")
print(f"Base: {BASE_MODEL_NAME}")
print(f"Output: {OUTPUT_PATH}")

# 1. Initialize Base Model
print("Loading base model...")
model = HypencoderDualEncoder.from_pretrained(BASE_MODEL_NAME)

# 2. Configure LoRA
# Values from my_configs/hypencoder.6_layer_lora.yaml
print("Applying LoRA config...")
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "key", "value", "dense"], 
)

# Apply to query encoder
model.query_encoder.transformer = get_peft_model(model.query_encoder.transformer, peft_config)
# Apply to passage encoder (shared)
model.passage_encoder.transformer = model.query_encoder.transformer

# 3. Load Trained Weights
print(f"Loading state dict from {CHECKPOINT_PATH}...")
state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"), map_location="cpu")

# Load state dict
# strict=False is safer given we might have extra keys or missing keys in some edge cases, 
# but specifically we expect the LoRA keys + the HyperHead keys to match.
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Missing keys: {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")

if len(missing) > 0:
    print("WARNING: Missing keys (might be okay if just unused config params):")
    for k in missing[:5]: print(f" - {k}")

# 4. Merge and Unload
print("Merging LoRA adapters...")
model.query_encoder.transformer = model.query_encoder.transformer.merge_and_unload()
# Passage encoder is shared reference, so it should be merged too.
# But just to be safe/sane about the object reference:
if model.passage_encoder.transformer is not model.query_encoder.transformer:
     print("Passage encoder was not shared reference?? Merging separately...")
     model.passage_encoder.transformer = model.passage_encoder.transformer.merge_and_unload()

# 5. Save
print(f"Saving merged model to {OUTPUT_PATH}...")
model.save_pretrained(OUTPUT_PATH)
# Also copy tokenizer
print("Copying tokenizer...")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.save_pretrained(OUTPUT_PATH)

print(" Done!")
