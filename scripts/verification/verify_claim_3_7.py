
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hypencoder_cb.modeling.hypencoder import Hypencoder, HypencoderConfig

def verify_claim_3_7():
    print("Verifying Claim 3.7: Hypencoder can emulate a Bi-Encoder")

    # 1. Setup Bi-Encoder Configuration
    # Ideally, a Bi-Encoder computes Score = q . d
    # Hypencoder generates a Q-Net. If Q-Net is a single linear layer with weight W,
    # then Score = d . W + b.
    # If W = q and b = 0, we have Score = d . q, which is the Bi-Encoder.
    
    # We configure Hypencoder to generate a 1-layer Q-Net (Input -> Output)
    dim = 768
    config = HypencoderConfig(
        model_name_or_path="bert-base-uncased", # Dummy, we won't use the transformer forward pass
        freeze_transformer=True,
        base_encoder_output_dim=dim,
        converter_kwargs={
            "vector_dimensions": [dim, 1], # Input 768, Output 1. No hidden layers.
            "activation_type": "relu", # Hypencoder enforces this locally in the converter, but for 1 layer it's just Linear
            "do_layer_norm": False,
            "do_residual": False,
            "do_dropout": False
        }
    )

    model = Hypencoder(config)
    
    # Hypencoder flow:
    # 1. Transformer -> last_hidden_state (Batch, Seq, Dim)
    # 2. _get_weights_and_biases -> matrices, vectors
    # 3. Models built from matrices/vectors
    # 4. Score
    
    # We want to show that we can choose parameters such that matrices[0] approx q.
    
    # Let's inspect _get_weights_and_biases
    # keys = key_projection(last_hidden_state)
    # values = value_projection(last_hidden_state)
    # weights = scaled_dot_product_attention(query, keys, values) ...
    # weights_final = WeightProj( LayerNorm( ReLU( weights ) ) ) + BaseMatrix
    
    # We want weights_final approx q.
    # Where q represents the query embedding.
    # Let's assume the Transformer output IS the query embedding q (e.g. CLS token).
    # We'll mock the transformer output.
    
    batch_size = 10
    seq_len = 1 # Simplified: Query is 1 token (or pooled)
    
    # Random query embeddings (Target weights)
    # Note: Because Hypencoder uses ReLU + LayerNorm, it can only generate weights
    # that lie in the range of WeightProj( LayerNorm( Positive(x) ) ).
    # If WeightProj is Identity, then weights must be LayerNormed.
    # So we should expect Hypencoder to emulate a NORMALIZED Bi-Encoder (Cosine Similarity-ish),
    # or a Bi-Encoder with specific weight constraints.
    # However, WeightProj is a Linear layer, so it can invert LayerNorm scaling if needed 
    # (though centering is harder to undo perfectly for all q without bias).
    
    # Let's try to set parameters to make it an IDENTITY generator.
    
    # Disable projections initially
    with torch.no_grad():
        # 1. Attention Mechanism
        # We want Attention to output the query embedding itself.
        # Attn(Q, K, V) = Softmax(Q K^T) V
        # If we have 1 sequence element, Softmax is 1.0.
        # So Attn output is V.
        # We want V = q.
        # key_projections: Don't care (since softmax is 1.0 for len=1)
        # value_projections: Identity mapping from input (dim) to internal dim (dim).
        
        # Hypencoder has a list of projections. We only have 1 weight matrix to generate.
        layer_idx = 0 
        
        # Set Value Projection to Identity
        model.value_projections[layer_idx].weight.copy_(torch.eye(dim))
        model.value_projections[layer_idx].bias.fill_(0.0)
        
        # 2. Post-Attention Processing
        # weights = V (which is q)
        # Process: WeightProj( LayerNorm( ReLU( weights ) ) ) + BaseMatrix
        
        # Problem: ReLU(q) destroys negative values of q.
        # This implies exact reconstruction of arbitrary q is impossible if q has negatives.
        # UNLESS we use bias/base matrix to shift it?
        # Or, maybe the "claim" relies on q being non-negative? (ReLU output of previous layer?)
        # Or maybe the paper implies "Universal Approximation" capability?
        
        # For this verification, let's assume q is positive (e.g., Output of a ReLU layer).
        
        # Set Weight Hyper Projection to Identity
        # But wait, there is LayerNorm. 
        # LN(x) = (x - mean) / std.
        # To undo LN, WeightProj needs to know mean/std, which are dynamic. 
        # So it cannot undo it perfectly for dynamic inputs.
        
        # However, we can set BaseMatrix to 0.
        model.hyper_base_matrices[layer_idx].data.fill_(0.0)
        
        # So essentially, Hypencoder generates W = WeightProj( LN( ReLU(q) ) ).
        # This means W is a linear transform of the normalized query.
        # This is equivalent to a Bi-Encoder where the query encoder ends with LayerNorm + Linear + ReLU.
        
        # Let's test precisely this equivalence.
        # Target Model: Bi-Encoder with Q' = WeightProj( LN( ReLU( q_raw ) ) )
        # Hypencoder Model: Should match Target Model.
        
        # Initialize WeightProj randomly (or identity)
        nn.init.eye_(model.weight_hyper_projection[layer_idx].weight)
        nn.init.zeros_(model.weight_hyper_projection[layer_idx].bias)
        
    # --- SYNTHETIC TEST ---
    
    # Mock Input: Query Embeddings
    q_raw = torch.randn(batch_size, seq_len, dim) 
    # Ensure they pass through ReLU "nicely" or just let ReLU do its thing.
    
    # Mock Input: Document Embeddings
    # Shape: (Batch, NumDocs, Dim)
    # Note: Hypencoder converter expects inputs to be (Batch, NumDocs, Dim)
    num_docs = 5
    docs = torch.randn(batch_size, num_docs, dim)
    
    # --- HYPENCODER FORWARD ---
    
    # 1. Get weights manually to mimic internal flow (we can also call model._get_weights_and_biases)
    # We need a dummy mask
    mask = torch.ones(batch_size, seq_len)
    
    # We need to hook the transformer to return q_raw
    # Actually, we can just call _get_weights_and_biases with q_raw assuming it's the hidden state
    matrices, vectors = model._get_weights_and_biases(q_raw, mask)
    
    # matrices[0] should be our generated weight W. Shape: (Batch, Dim, 1) or similar?
    print(f"Config Converter Kwargs: {config.converter_kwargs}")
    
    # Hypencoder returns (Batch, OutDim, InDim).
    # Here In=768, Out=1. So (Batch, 1, 768).
    generated_weight = matrices[0] 
    # generated_bias = vectors[0] # Bias is empty for 1-layer 
    
    # Now run the Q-Net logic manually or via converter
    # Q-Net logic for 1 layer: y = x @ W^T
    # But wait, generated_weight is (Batch, Out, In).
    # Standard Linear layer weight W is (Out, In).
    # x @ W^T -> (Batch, In) @ (Batch, In, Out) -> (Batch, Out).
    # generated_weight.transpose(1, 2) is (Batch, In, Out).
    # So using transpose IS correct if generated_weight is (Batch, Out, In).
    
    # Let's check generated_weight shape.
    # If it is (Batch, 768, 1), then In=1, Out=768? 
    # Or In=768, Out=1 ?
    # weight_shapes[0] = (Input, Output).
    # hypencoder.py: for in, out in shapes:
    #   weight_query_embeddings: (1, out, in)
    #   attn results: (Batch, out, in)
    # So generated_weight is (Batch, Output, Input).
    
    print(f"Docs shape: {docs.shape}")
    print(f"Generated Weight shape (Batch, Out, In): {generated_weight.shape}")
    
    # Assert expected shape
    if generated_weight.shape[1] != 1 or generated_weight.shape[2] != dim:
        print(f"WARNING: generated_weight shape {generated_weight.shape} is unexpected for [dim, 1] config.")
        # If it is (Batch, 768, 768), adjust to prove usage works anyway?
        # But previous debug showed (1, 1, 768) param shapes.
        # So generated_weight MUST be (Batch, 1, 768).
        if generated_weight.shape[1] == dim:
             print("Assuming generated_weight is (Batch, Dim, Dim).")
    
    generated_weight_T = generated_weight.transpose(1, 2)
    print(f"Generated Weight Transposed (Batch, In, Out): {generated_weight_T.shape}")
    
    hypencoder_scores = torch.einsum("bid,bdh->bih", docs, generated_weight_T)
    print(f"Hypencoder Scores shape: {hypencoder_scores.shape}")
    
    # Assert score shape
    if hypencoder_scores.shape[-1] != 1:
        print(f"WARNING: Hypencoder score dim is {hypencoder_scores.shape[-1]}. Expected 1.")
        # If it is 768, take the first dimension? Or sum?
        # But for Bi-Encoder, we expect 1 score.
        pass
    
    # --- TARGET BI-ENCODER FORWARD ---
    
    # Calculate what q_transformed should be according to the architecture
    # Flow: q -> Key/Val Proj -> Attn -> ReLU -> LN -> HyperProj -> Add Base -> Transform
    
    # With our settings:
    # Keys/Vals: Identity(q) (Sequence len 1 so Attn is just Identity passing values)
    # Attn Out: q
    # Layer 1 Processing:
    # x = ReLU(q)
    # x = LayerNorm(x)
    # x = HyperProj(x) (Identity)
    # x = x + Base (0)
    # W = x.transpose()
    
    # Replicate explicitly:
    q_squeezed = q_raw.squeeze(1) # (Batch, Dim)
    
    # Apply the FIXED Hypencoder transformations
    # Note: These are hardcoded in Hypencoder._get_weights_and_biases
    x = F.relu(q_squeezed)
    x = F.layer_norm(x, (dim,))
    
    # Our manual setting of HyperProj is Identity and Base is 0
    q_transformed = x # (Batch, Dim)
    
    # Target Score = d . q_transformed
    # d: (Batch, NumDocs, Dim)
    # q encoded as weight: (Batch, 1, Dim) (from x) -> Transposed in Q-Net to (Batch, Dim, 1)
    
    # Target: d @ q_transformed^T (conceptually)
    
    bi_encoder_scores = torch.einsum("bid,bd->bi", docs, q_transformed).unsqueeze(-1)
    
    # Check match
    diff = (hypencoder_scores - bi_encoder_scores).abs()
    print(f"Mean Difference: {diff.mean().item()}")
    print(f"Max Difference: {diff.max().item()}")
    
    if diff.max() < 1e-4: # Tolerance for float32/LayerNorm epsilons
        print("SUCCESS: Hypencoder scores match the simulated Bi-Encoder exactly (accounting for LN/ReLU).")
        print("Result: Hypencoder implements 'Bi-Encoder with LayerNormed-ReLU Queries'.")
    else:
        print("FAILURE: Scores do not match.")
        print("Hypencoder output sample:", hypencoder_scores[0,0])
        print("Target output sample:", bi_encoder_scores[0,0])

if __name__ == "__main__":
    verify_claim_3_7()
