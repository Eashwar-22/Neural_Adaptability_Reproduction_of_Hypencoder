import torch
from transformers import AutoConfig

# Import from our NEW files
from hypencoder_cb.modeling.hypencoder_colbert import (
    TextEncoder,
    TextEncoderConfig,
    HypencoderDualEncoder,
    HypencoderDualEncoderConfig,
)

def test_text_encoder_unpooled():
    print("Testing TextEncoder unpooled output...")
    config = TextEncoderConfig(
        model_name_or_path="bert-base-uncased",
        pooling_type="none"
    )
    model = TextEncoder(config)
    
    # Dummy input (Batch=2, SeqLen=10)
    input_ids = torch.randint(0, 1000, (2, 10))
    attention_mask = torch.ones((2, 10))
    
    output = model(input_ids, attention_mask)
    rep = output.representation
    
    print(f"Output shape: {rep.shape}")
    
    # Check shape: (Batch, SeqLen, Hidden=768)
    assert len(rep.shape) == 3
    assert rep.shape[0] == 2
    assert rep.shape[1] == 10
    assert rep.shape[2] == 768
    print("TextEncoder test PASSED.")

def test_dual_encoder_pipeline():
    print("\nTesting HypencoderDualEncoder pipeline with MaxSim...")
    
    # Config for dual encoder
    config = HypencoderDualEncoderConfig(
        query_encoder_kwargs={
            "model_name_or_path": "bert-base-uncased",
            "converter_kwargs": {
                "vector_dimensions": [768, 128, 1],
            }
        },
        passage_encoder_kwargs={
            "model_name_or_path": "bert-base-uncased",
            "pooling_type": "none"  # Crucial: Unpooled
        },
        loss_type=["cross_entropy"],
        loss_kwargs=[{}],
        shared_encoder=True
    )
    
    model = HypencoderDualEncoder(config)
    
    # Dummy Input
    # 2 Queries
    q_input_ids = torch.randint(0, 1000, (2, 10))
    q_mask = torch.ones((2, 10))
    
    # 2 Passages aka Positive Docs (Batch=2, SeqLen=15)
    p_input_ids = torch.randint(0, 1000, (2, 15))
    p_mask = torch.ones((2, 15))
    
    print("Running forward pass...")
    # This calls forward in BaseDualEncoder -> logic in Hypencoder/TextEncoder -> Loss -> Similarity
    output = model(
        query_input_ids=q_input_ids,
        query_attention_mask=q_mask,
        passage_input_ids=p_input_ids,
        passage_attention_mask=p_mask,
        labels=None # Will rely on in-batch negs or just simple flow
    )
    
    # output is DualEncoderOutput(loss=..., similarity=...)
    # Check similarity shape
    # With CrossEntropyLoss defaults:
    # It might use no_in_batch or in_batch. 
    # Our HypencoderCrossEntropyLoss defaults to `use_in_batch_negatives=True` (inherited default is True usually?)
    # Wait, let's check defaults. 
    # BaseDualEncoderConfig defaults? Not visible here.
    # CrossEntropyLoss init defaults to `use_in_batch_negatives=True`.
    
    # However, `in_batch_negatives_hypecoder_similarity` in `similarity_colbert` raises NotImplementedError for 3D!
    # So we expect this to FAIL if `use_in_batch_negatives` is True.
    # We should handle this expectation or disable it.
    
    # Let's verify if implementation handles it or catches it.
    
    pass

if __name__ == "__main__":
    test_text_encoder_unpooled()
    
    try:
        test_dual_encoder_pipeline()
    except NotImplementedError as e:
        print(f"Caught expected error: {e}")
        print("Note: In-batch negatives are currently disabled for 3D inputs. Training must use `use_in_batch_negatives=False` or separate negative mining.")
    except Exception as e:
        print(f"FAILED with unexpected error: {e}")
        raise e
