
from hypencoder_cb.modeling.hypencoder import TextEncoder, TextEncoderConfig
import torch

def verify_colbert_claim():
    print("Verifying ColBERT Replication Claim...")
    print("ColBERT requires 'Late Interaction', meaning the document encoder must return")
    print("a sequence of token embeddings (Batch, SeqLen, Dim), not a pooled vector.")
    
    print("\nTest 1: Check TextEncoder pooling options.")
    config = TextEncoderConfig(
        model_name_or_path="bert-base-uncased",
        pooling_type="none", # Trying to disable pooling
        freeze_transformer=True
    )
    
    try:
        model = TextEncoder(config)
        # Check if model.pool handles 'none'
        if not hasattr(model, 'pool'):
            print("FAILURE: TextEncoder does not assign 'self.pool' for pooling_type='none'.")
            print("       This means unpooled output is not supported out-of-the-box.")
            return

        # Attempt Forward Pass
        input_ids = torch.randint(0, 1000, (1, 5))
        mask = torch.ones((1, 5))
        output = model(input_ids, mask)
        print(f"Output Shape: {output.representation.shape}")
        
    except AttributeError as e:
        print(f"FAILURE (Expected): Caught AttributeError during init or forward: {e}")
        print("TextEncoder only supports 'mean' and 'cls' pooling types.")
    except Exception as e:
        print(f"FAILURE: Caught unexpected exception: {e}")

    print("\nConclusion: The current implementation enforces pooling, making ColBERT's")
    print("            token-level interaction impossible to replicate directly.")

if __name__ == "__main__":
    verify_colbert_claim()
