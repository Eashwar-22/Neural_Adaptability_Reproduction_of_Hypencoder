import os
from typing import Optional

import fire
from datasets import load_dataset, load_from_disk
from omegaconf import OmegaConf
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer import TRAINER_STATE_NAME

# CHANGED: Import from hypencoder_colbert
from hypencoder_cb.modeling.hypencoder_colbert import (
    HypencoderDualEncoder,
    HypencoderDualEncoderConfig,
)
from hypencoder_cb.modeling.shared import BaseDualEncoderConfig
from hypencoder_cb.train.args import (
    HypencoderDataConfig,
    HypencoderModelConfig,
    HypencoderTrainerConfig,
    HypencoderTrainingConfig,
)
from hypencoder_cb.train.data_collator import GeneralDualEncoderCollator
from peft import get_peft_model, LoraConfig, TaskType

DEFAULT_CACHE_DIR = os.environ.get(
    "HYPENCODER_CACHE_DIR", os.path.expanduser("~/.cache/hypencoder")
)

import torch

_original_torch_load = torch.load
def _unsafe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _unsafe_torch_load


def load_model(model_config: HypencoderModelConfig):
    config_cls_lookup = {
        "hypencoder": HypencoderDualEncoderConfig,
        "biencoder": BaseDualEncoderConfig,
    }

    model_cls_lookup = {
        "hypencoder": HypencoderDualEncoder,
        # "biencoder": TextDualEncoder, # Not supported in this script
    }

    config_cls = config_cls_lookup[model_config.model_type]
    model_cls = model_cls_lookup[model_config.model_type]

    config = config_cls(
        query_encoder_kwargs=OmegaConf.to_container(
            model_config.query_encoder_kwargs
        ),
        passage_encoder_kwargs=OmegaConf.to_container(
            model_config.passage_encoder_kwargs
        ),
        loss_type=OmegaConf.to_container(model_config.loss_type),
        loss_kwargs=OmegaConf.to_container(model_config.loss_kwargs),
        shared_encoder=model_config.shared_encoder,
    )

    if model_config.checkpoint_path is not None:
        model = model_cls.from_pretrained(
            model_config.checkpoint_path, config=config
        )
    else:
        model = model_cls(config)

    if model_config.use_lora:
        print(f"Wrapping transformer backbone with LoRA (r={model_config.lora_r})...")
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
        
        model.query_encoder.transformer.print_trainable_parameters()

    return model


def load_data(data_config: HypencoderDataConfig):
    cache_dir = os.environ.get("HF_HOME", DEFAULT_CACHE_DIR)

    if (data_config.training_data_jsonl is None) == (
        data_config.training_huggingface_dataset is None
    ):
        raise ValueError(
            "Must specify either training_data_jsonl or"
            " training_huggingface_dataset"
        )

    if (
        data_config.validation_data_jsonl is not None
        and data_config.validation_huggingface_dataset is not None
    ):
        raise ValueError(
            "Cannot specify both validation_data_jsonl and"
            " validation_huggingface_dataset"
        )

    if data_config.training_huggingface_dataset is not None:
        training_data = load_dataset(
            data_config.training_huggingface_dataset,
            split=data_config.training_data_split,
            cache_dir=cache_dir,
        )
    elif data_config.training_data_jsonl is not None:
        dataset_path = data_config.training_data_jsonl + ".dataset"
        if os.path.exists(dataset_path):
            print(f"Loading pre-saved dataset from disk: {dataset_path}")
            training_data = load_from_disk(dataset_path)
        else:
            print(f"Loading from JSONL and saving to disk: {dataset_path}")
            training_data = load_dataset(
                "json",
                data_files=data_config.training_data_jsonl,
                split=data_config.training_data_split,
                cache_dir=cache_dir,
            )
            if os.environ.get("RANK", "0") == "0":
                print(f"Saving dataset to disk at {dataset_path}...")
                training_data.save_to_disk(dataset_path)
                print("Dataset saved.")

    validation_data = None
    if data_config.validation_huggingface_dataset is not None:
        training_data = load_dataset(
            data_config.validation_huggingface_dataset,
            split=data_config.validation_data_split,
            cache_dir=cache_dir,
        )
    elif data_config.validation_data_jsonl is not None:
        training_data = load_dataset(
            "json",
            data_files=data_config.validation_data_jsonl,
            split=data_config.validation_data_split,
            cache_dir=cache_dir,
        )

    return training_data, validation_data


def get_collator(
    data_config: HypencoderDataConfig,
    trainer_config: HypencoderTrainerConfig,
    tokenizer,
):
    return GeneralDualEncoderCollator(
        tokenizer=tokenizer,
        num_negatives_to_sample=data_config.num_negatives_to_sample,
        positive_filter=data_config.positive_filter_type,
        positive_filter_kwargs=data_config.positive_filter_kwargs,
        positive_sampler="random",
        negative_sampler="random",
        num_positives_to_sample=data_config.num_positives_to_sample,
        label_key=data_config.label_key,
        query_padding_mode="longest",
    )


def load_tokenizer(model_config: HypencoderModelConfig):
    return AutoTokenizer.from_pretrained(
        model_config.tokenizer_pretrained_model_name_or_path
    )


def train_model(cfg: HypencoderTrainingConfig):

    print(cfg)
    resume_from_checkpoint = cfg.trainer_config.resume_from_checkpoint

    training_data, validation_data = load_data(cfg.data_config)
    tokenizer = load_tokenizer(cfg.model_config)
    model = load_model(cfg.model_config)
    collator = get_collator(cfg.data_config, cfg.trainer_config, tokenizer)

    train_arguments_kwargs = None
    hf_trainer_config = cfg.trainer_config.hf_trainer_config

    if OmegaConf.is_config(hf_trainer_config):
        train_arguments_kwargs = OmegaConf.to_container(hf_trainer_config)
    else:
        train_arguments_kwargs = hf_trainer_config.__dict__

    if "local_rank" not in train_arguments_kwargs:
         train_arguments_kwargs["local_rank"] = int(os.environ.get("LOCAL_RANK", -1))

    training_args = TrainingArguments(
        **train_arguments_kwargs,
    )

    if training_args.save_on_each_node:
        print("WARNING: save_on_each_node was True. Forcing to False to prevent race conditions.")
        training_args.save_on_each_node = False

    class CustomTrainer(Trainer):
        def _save_checkpoint(self, model, trial, metrics=None):
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            run_dir = self.args.output_dir
            output_dir = os.path.join(run_dir, checkpoint_folder)
            
            os.makedirs(output_dir, exist_ok=True)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            
            self.save_model(output_dir, _internal_call=False)
            
            if not self.args.save_only_model:
                self._save_optimizer_and_scheduler(output_dir)
                self._save_rng_state(output_dir)
                
            if self.args.should_save:
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
                
            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)
                
            if self.args.should_save:
                self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=training_data,
        eval_dataset=validation_data,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("Starting training")
    if resume_from_checkpoint is True:
        if not os.path.exists(training_args.output_dir) or not any(
            [
                "checkpoint" in name
                for name in os.listdir(training_args.output_dir)
            ]
        ):
            resume_from_checkpoint = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    print(f"Training finished. Saving final model to {training_args.output_dir}...")
    trainer.save_model()
    trainer.save_state()


def run_training(config_path: Optional[str] = None) -> None:
    schema = OmegaConf.structured(HypencoderTrainingConfig)

    if config_path is not None:
        config = OmegaConf.load(config_path)
        config = OmegaConf.merge(schema, config)
    else:
        config = schema

    train_model(config)


if __name__ == "__main__":
    fire.Fire(run_training)
