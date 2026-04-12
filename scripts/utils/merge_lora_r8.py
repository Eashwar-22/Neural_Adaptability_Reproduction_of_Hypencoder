
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
    
    if not os.path.exists(config_yaml_path):
        raise FileNotFoundError(f"Config not found: {config_yaml_path}")
    
    full_config = OmegaConf.load(config_yaml_path)
    model_config = full_config.model_config
    
    print("Instantiating Base HypencoderDualEncoder...")
    config = HypencoderDualEncoderConfig(
        query_encoder_kwargs=OmegaConf.to_container(model_config.query_encoder_kwargs),
        passage_encoder_kwargs=OmegaConf.to_container(model_config.passage_encoder_kwargs),
        loss_type=OmegaConf.to_container(model_config.loss_type),
        loss_kwargs=OmegaConf.to_container(model_config.loss_kwargs),
        shared_encoder=model_config.shared_encoder,
    )
    
    model = HypencoderDualEncoder(config)
    
    if model_config.use_lora:
        print(f"Wrapping transformer backbone with LoRA (r={model_config.lora_r}) to match checkpoint structure...")
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
            target_modules=["query", "key", "value", "dense"], 
        )
        model.query_encoder.transformer = get_peft_model(model.query_encoder.transformer, peft_config)
        if model_config.shared_encoder:
            model.passage_encoder.transformer = model.query_encoder.transformer
        else:
             model.passage_encoder.transformer = get_peft_model(model.passage_encoder.transformer, peft_config)
    else:
        raise ValueError("Config says use_lora=False. This script is for merging LoRA models.")

    print(f"Loading state dict from {checkpoint_path}/pytorch_model.bin ...")
    state_dict_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if not os.path.exists(state_dict_path):
         raise FileNotFoundError(f"State dict not found: {state_dict_path}")
         
    import torch
    _original_torch_load = torch.load
    def _unsafe_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _unsafe_torch_load
    
    state_dict = torch.load(state_dict_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Load results - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    torch.load = _original_torch_load

    print("Merging adapters...")
    model.query_encoder.transformer = model.query_encoder.transformer.merge_and_unload()
    if model_config.shared_encoder:
        model.passage_encoder.transformer = model.query_encoder.transformer
    else:
        model.passage_encoder.transformer = model.passage_encoder.transformer.merge_and_unload()

    print(f"Saving merged model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_pretrained_model_name_or_path)
    tokenizer.save_pretrained(output_path)
    print("Done!")

if __name__ == "__main__":
    # Settings for LoRA r8
    CONFIG_YAML = "my_configs/hypencoder.6_layer_lora.yaml"
    CHECKPOINT_DIR = "checkpoints/hypencoder.6_layer_lora/checkpoint-4000"
    OUTPUT_DIR = "checkpoints/hypencoder.6_layer_lora/merged_checkpoint-4000_fixed"
    
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"Error: Checkpoint not found at {CHECKPOINT_DIR}")
        exit(1)
        
    merge_lora_weights(CHECKPOINT_DIR, OUTPUT_DIR, CONFIG_YAML)
