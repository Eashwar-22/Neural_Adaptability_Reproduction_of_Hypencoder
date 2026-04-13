
import torch
import random
from transformers import AutoTokenizer
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder, HypencoderConfig
from hypencoder_cb.train.data_collator import GeneralDualEncoderCollator

def verify_dropout():
    print("--- Verifying Query Dropout ---")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    collator = GeneralDualEncoderCollator(
        tokenizer=tokenizer,
        num_negatives_to_sample=1,
        query_dropout_prob=0.5,
        positive_filter="type",
        positive_filter_kwargs={"positive_type": "pos"}
    )
    
    # Dummy feature
    query_text = "this is a test query for dropout verification"
    tokenized = tokenizer.encode(query_text, add_special_tokens=True) # [CLS, ... tokens ..., SEP]
    print(f"Original tokens: {tokenized} (Len: {len(tokenized)})")
    
    features = [{
        "query": {"tokenized_content": tokenized},
        "items": [{"tokenized_content": [101, 200, 102], "score": 1, "type": "pos"}, {"tokenized_content": [101, 300, 102], "score": 0, "type": "neg"}]
    }]
    
    # Run multiple times to see variation
    for i in range(3):
        batch = collator(features)
        q_ids = batch["query_input_ids"][0].tolist()
        # Remove padding
        q_ids = [t for t in q_ids if t != tokenizer.pad_token_id]
        print(f"Run {i}: {q_ids} (Len: {len(q_ids)})")
        
    print("Dropout verified if lengths vary and are < original.")

def verify_regularization():
    print("\n--- Verifying Q-Net Regularization ---")
    config = HypencoderConfig(
        model_name_or_path="google-bert/bert-base-uncased",
        qnet_regularization_lambda=1.0, # Strong regularization
        converter_kwargs={
            "vector_dimensions": [768, 768, 768, 768, 768, 768, 768, 1],
            "activation_type": "relu",
            "do_residual_on_last": False
        }
    )
    # We need to initialize Hypencoder directly to test forward, or DualEncoder
    # Let's use internal Hypencoder to keep it simple
    from hypencoder_cb.modeling.hypencoder import Hypencoder
    model = Hypencoder(config)
    
    # Dummy input
    input_ids = torch.tensor([[101, 200, 201, 102]])
    attention_mask = torch.tensor([[1, 1, 1, 1]])
    
    output = model(input_ids, attention_mask)
    
    print(f"Regularization Loss: {output.loss}")
    if output.loss > 0:
        print("Regularization verified (Loss > 0).")
    else:
        print("Regularization FAILED (Loss = 0). Check implementation.")

if __name__ == "__main__":
    verify_dropout()
    verify_regularization()
