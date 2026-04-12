
import os
import sys
import torch
from omegaconf import OmegaConf
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

# Add project root to path
sys.path.append(os.getcwd())

from hypencoder_cb.modeling.hypencoder import (
    HypencoderDualEncoder,
    HypencoderDualEncoderConfig,
)
from hypencoder_cb.train.args import HypencoderModelConfig

def merge_lora_weights(checkpoint_path, output_path, config_yaml_path):
    print(f"Loading configuration from {checkpoint_path} and {config_yaml_path}...")
    
    # 1. Load the Training Config (to get Model Architecture & LoRA params)
    # We need to know d_model, lora_r, etc.
    if not os.path.exists(config_yaml_path):
        raise FileNotFoundError(f"Config not found: {config_yaml_path}")
    
    full_config = OmegaConf.load(config_yaml_path)
    model_config = full_config.model_config
    
    # 2. Instantiate the Base HypencoderDualEncoder
    print("Instantiating Base HypencoderDualEncoder...")
    # Map config to class args (similar to train.py)
    # We rely on the YAML config being correct for the model structure
    
    config = HypencoderDualEncoderConfig(
        query_encoder_kwargs=OmegaConf.to_container(model_config.query_encoder_kwargs),
        passage_encoder_kwargs=OmegaConf.to_container(model_config.passage_encoder_kwargs),
        loss_type=OmegaConf.to_container(model_config.loss_type),
        loss_kwargs=OmegaConf.to_container(model_config.loss_kwargs),
        shared_encoder=model_config.shared_encoder,
    )
    
    # Initialize random model first
    model = HypencoderDualEncoder(config)
    
    # 3. Apply LoRA Wrapping (Empty/Random) to match structure
    # This prepares the model to receive the state_dict which contains LoRA keys
    if model_config.use_lora:
        print(f"Wrapping transformer backbone with LoRA (r={model_config.lora_r}) to match checkpoint structure...")
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False, # Must be False to load trainable params correctly if needed? Actually for loading state dict structure matters.
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
            target_modules=["query", "key", "value", "dense"], 
        )
        
        # Apply to query encoder's transformer
        model.query_encoder.transformer = get_peft_model(model.query_encoder.transformer, peft_config)
        
        # If shared, ensure passage encoder uses the same PEFT model
        if model_config.shared_encoder:
            model.passage_encoder.transformer = model.query_encoder.transformer
        else:
             model.passage_encoder.transformer = get_peft_model(model.passage_encoder.transformer, peft_config)
    else:
        raise ValueError("Config says use_lora=False. This script is for merging LoRA models.")

    # 4. Load the State Dict from the Checkpoint
    print(f"Loading state dict from {checkpoint_path}/pytorch_model.bin ...")
    state_dict_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if not os.path.exists(state_dict_path):
         raise FileNotFoundError(f"State dict not found: {state_dict_path}")
         
    # Use unsafe load because the file might rely on it (as seen in train.py workaround)
    import torch
    _original_torch_load = torch.load
    def _unsafe_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _unsafe_torch_load
    
    state_dict = torch.load(state_dict_path, map_location="cpu")
    
    # filter out 'vocab' related keys if they mismatch/cause issues, but usually fine.
    # Load into model
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Load results - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    if len(missing) > 0:
        print("Missing keys (first 10):", missing[:10])
    if len(unexpected) > 0:
        print("Unexpected keys (first 10):", unexpected[:10])
        
    # Restore torch.load
    torch.load = _original_torch_load

    # 5. Merge and Unload
    print("Merging adapters...")
    # We need to access the PeftModel directly.
    # model.query_encoder.transformer is the PeftModel (wrapped)
    
    # This edits the module in-place, replacing PeftModel with the base transformer with merged weights
    model.query_encoder.transformer = model.query_encoder.transformer.merge_and_unload()
    
    # Update passage encoder reference if shared
    if model_config.shared_encoder:
        model.passage_encoder.transformer = model.query_encoder.transformer
    else:
        model.passage_encoder.transformer = model.passage_encoder.transformer.merge_and_unload()

    print("Merge complete. Model is now standard architecture.")

    # 6. Save the Merged Model
    print(f"Saving merged model to {output_path}...")
    
    # Now we can save using the standard save_pretrained
    # This will save config.json and pytorch_model.bin (standard)
    model.save_pretrained(output_path)
    
    # Save tokenizer too
    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_pretrained_model_name_or_path)
    tokenizer.save_pretrained(output_path)
    
    print("Done!")

if __name__ == "__main__":
    # Settings
    CONFIG_YAML = "my_configs/hypencoder.6_layer_lora_r64_v2.yaml"
    CHECKPOINT_DIR = "checkpoints/hypencoder.6_layer_lora_r64_v2/checkpoint-4500"
    OUTPUT_DIR = "checkpoints/hypencoder.6_layer_lora_r64_v2/merged_checkpoint-4500"
    
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"Error: Checkpoint not found at {CHECKPOINT_DIR}")
        exit(1)
        
    merge_lora_weights(CHECKPOINT_DIR, OUTPUT_DIR, CONFIG_YAML)
