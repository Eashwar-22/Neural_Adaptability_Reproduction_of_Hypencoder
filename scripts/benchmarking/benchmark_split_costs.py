
import time
import torch
import torch.nn as nn
from transformers import AutoConfig
from hypencoder_cb.modeling.hypencoder_colbert import Hypencoder, HypencoderConfig

def benchmark_split_costs():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Efficiency Benchmark on {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'} ===")
    
    # --- Configuration ---
    # Matches hypencolbert_multihead_full.yaml
    DIM = 768
    # Config: [768, 768, 768, 768, 768, 768, 768, 32] -> Input + 7 layers
    # RepeatedDenseBlockConverter logic:
    # weight_shapes[i] = (vector_dimensions[i-1], vector_dimensions[i])
    # The config list INCLUDES the input dim as first element? 
    # Let's check the yaml: vector_dimensions: [768, 768, ..., 32]
    # In `q_net.py`: for i in range(1, len(vector_dimensions)): ...
    # So yes, first element is input.
    VECTOR_DIMS = [768, 768, 768, 768, 768, 768, 768, 32]
    
    SEQ_LEN = 180 # Avg doc length
    BATCH_QUERIES = 64 # Realistic query batch size
    DTYPE = torch.bfloat16
    
    print(f"Config: Dim={DIM}, Vector Dims={VECTOR_DIMS}, SeqLen={SEQ_LEN}, Dtype={DTYPE}")

    # --- Setup Model ---
    config = HypencoderConfig(
        model_name_or_path="google-bert/bert-base-uncased",
        base_encoder_output_dim=DIM,
        converter_kwargs={
            "vector_dimensions": VECTOR_DIMS,
            "activation_type": "relu",
            "do_residual_on_last": False
        }
    )
    
    # Initialize Model (Weights are random, fine for timing)
    model = Hypencoder(config).to(device).to(DTYPE)
    model.eval()
    
    # Mock Token IDs
    # (Batch, Seq)
    input_ids = torch.randint(0, 30522, (BATCH_QUERIES, 32), device=device)
    attention_mask = torch.ones((BATCH_QUERIES, 32), device=device)


    # --- 1. Pre-computation Benchmark (Query -> Weights) ---
    print("\n--- Phase 1: Pre-computation (Query -> Q-Net Weights) ---")
    
    # Warmup
    print("Warming up Generator...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids, attention_mask)
    torch.cuda.synchronize()
    
    # Timing
    iters = 50
    start = time.time()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(input_ids, attention_mask)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_gen_time_ms = ((end - start) / iters) * 1000
    gen_time_per_query_ms = avg_gen_time_ms / BATCH_QUERIES
    
    print(f"Batch Generation Time ({BATCH_QUERIES} queries): {avg_gen_time_ms:.4f} ms")
    print(f"Per-Query Pre-computation Cost: {gen_time_per_query_ms:.4f} ms")
    
    
    # --- 2. Interaction Benchmark (Weights + Docs -> Scores) ---
    print("\n--- Phase 2: Interaction Cost (Apply Q-Net to Documents) ---")
    
    N_DOCS_LIST = [1000, 10000] 
    
    # Get a sample generated Q-Net module
    # We take the first query's module for this test.
    # The output of `model` contains `representation` which IS the module.
    # But wait, `Hypencoder.forward` returns `HypencoderOutput(representation=models)`.
    # `models` is a `NoTorchSequential` object.
    # It wraps `BatchLinear` layers.
    # `BatchLinear` expects input `(Batch, Items, Dim)`.
    # It computes `(Batch, Items, Dim) x (Batch, Dim, Out)`.
    # If we want to check 1 Query vs N Docs:
    # We can fake it by setting Batch=1.
    
    # Generate for Batch=1
    input_ids_1 = input_ids[0:1] # (1, 32)
    mask_1 = attention_mask[0:1]
    
    with torch.no_grad():
        out = model(input_ids_1, mask_1)
        q_net_module = out.representation # NoTorchSequential
        
    for n_docs in N_DOCS_LIST:
        print(f"\nScanning {n_docs} Documents (Seq={SEQ_LEN})...")
        
        # Doc Embeddings: (Batch=1, N_Docs * Seq, Dim)
        # We flatten documents into the "Items" dimension
        total_items = n_docs * SEQ_LEN
        docs = torch.randn(1, total_items, DIM, device=device, dtype=DTYPE)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                 _ = q_net_module(docs)
        torch.cuda.synchronize()
        
        start = time.time()
        inner_iters = 10
        with torch.no_grad():
            for _ in range(inner_iters):
                # 1. Forward Pass (Q-Net)
                # Output: (1, TotalItems, 32)
                out_tensor = q_net_module(docs)
                
                # Reshape to (1, N_Docs, Seq, 32)
                out_tensor = out_tensor.view(1, n_docs, SEQ_LEN, 32)
                
                # 2. MaxSim (Pool over Seq) -> (1, N_Docs, 32)
                out_tensor = out_tensor.max(dim=2).values
                
                # 3. Sum (Pool over Heads) -> (1, N_Docs)
                scores = out_tensor.sum(dim=2)
            
        torch.cuda.synchronize()
        end = time.time()
        
        avg_scan_time_ms = ((end - start) / inner_iters) * 1000
        print(f"Time to Scan {n_docs} Docs: {avg_scan_time_ms:.4f} ms")
        print(f"Latency per 1k Docs: {avg_scan_time_ms / (n_docs/1000):.4f} ms")


    # --- 3. Storage Analysis ---
    print("\n--- Phase 3: Storage Analysis ---")
    
    num_docs = 8_841_823 # MS MARCO
    
    # Bi-Encoder / Hypencoder (Pooled)
    # 1 Vector per Doc
    vec_size_bytes = DIM * 2 # FP16/BF16
    total_pooled_gb = (num_docs * vec_size_bytes) / 1e9
    
    # HypenColBERT (Unpooled)
    # SeqLen Vectors per Doc
    total_unpooled_gb = (num_docs * SEQ_LEN * vec_size_bytes) / 1e9
    
    # Quantized
    total_unpooled_int8_gb = (num_docs * SEQ_LEN * DIM * 1) / 1e9 
    
    print(f"MS MARCO ({num_docs:,} docs)")
    print(f"Pooled Index (1 vec/doc, BF16): {total_pooled_gb:.2f} GB")
    print(f"HypenColBERT Index ({SEQ_LEN} vecs/doc, BF16): {total_unpooled_gb:.2f} GB")
    print(f"HypenColBERT Index (Int8 Quantized): {total_unpooled_int8_gb:.2f} GB")
    print(f"Storage Increase Factor: {total_unpooled_gb / total_pooled_gb:.1f}x")

if __name__ == "__main__":
    benchmark_split_costs()
