
import time
import torch
from transformers import AutoConfig
from hypencoder_cb.modeling.hypencoder_colbert import HypencoderDualEncoder, TextEncoderConfig
from hypencoder_cb.modeling.q_net import RepeatedDenseBlockConverter

# Mock Config matching Multi-Head HypenColBERT
def benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. Setup Dummy Model/Data
    BATCH_DOCS = 2000
    BATCH_QUERIES = 100 # In the script, it loops queries in batches of 10 or 100 against the doc chunk.
    # The script `inference_hypencolbert.sh` says `query_batch_size 10`.
    BATCH_QUERIES_REAL = 10 
    
    SEQ_LEN = 180
    DIM = 768
    HEADS = 32
    
    print(f"Benchmarking Batch: Docs={BATCH_DOCS}, Queries={BATCH_QUERIES_REAL}, Seq={SEQ_LEN}, Dim={DIM}, Heads={HEADS}")
    
    # Mock Tensors
    # Docs: (2000, 180, 768)
    doc_embeddings = torch.randn(BATCH_DOCS, SEQ_LEN, DIM, device=device, dtype=torch.bfloat16)
    
    # Query Param Generator Mock (We just need the forward pass of the Q-Net)
    # A single Q-Net forward pass:
    # Input: (Docs*Seq, Dim) -> Output: (Docs*Seq, 32)
    # We do this for EACH query in the batch.
    
    # To be realistic, we need to create the actual Q-Net structure or a proxy.
    # Let's use a simple linear layer proxy for speed test, knowing Q-Net is MLP (Linear-ReLU-Linear).
    # Input: 768 -> Hidden -> 32
    
    # We will simulate the "Inner Loop" of `retrieve_colbert.py`
    # Warning: The actual Q-Net is created DYNAMICALLY.
    # But for speed, it's just a sequence of Matrix Multiplications.
    
    # Q-Net Structure from `hypencolbert_multihead.yaml`:
    # vector_dimensions: [768, 768, 768, ..., 32]
    # It's deep! 7 layers?
    # Let's assume 3 layers for a conservative check or check config.
    # Config says: [768, 768, 768, 768, 768, 768, 768, 32] -> 7 Layers!
    
    # So we need to run 7 Linear Layers for every token.
    
    layer_weights = []
    for _ in range(7):
        # Weight: (768, 768)
        w = torch.randn(768, 768, device=device, dtype=torch.bfloat16)
        layer_weights.append(w)
        
    last_w = torch.randn(768, 32, device=device, dtype=torch.bfloat16)
    
    # 2. Timing Loop
    iters = 10
    
    # Warmup
    torch.cuda.synchronize()
    
    start = time.time()
    
    for _ in range(iters):
        # For each Query in the Query Batch (10)
        # We broadcast the specific weights. 
        # In reality, `NoTorchSequential` does a batched matmul: (Batch, N, In) x (Batch, In, Out)
        
        # Let's simulate the operation:
        # Input: (1, N_Docs*Seq, 768) expanded to (10, N_Docs*Seq, 768)
        # Weights: (10, 768, 768)
        
        x = doc_embeddings.view(1, -1, DIM).expand(BATCH_QUERIES_REAL, -1, -1) # (10, 360000, 768)
        
        for w in layer_weights:
            # Matmul: (10, 360k, 768) x (10, 768, 768) -> (10, 360k, 768)
            # This is effectively a Batch MatMul
            # We simulate weights for the batch
            batch_w = w.unsqueeze(0).expand(BATCH_QUERIES_REAL, -1, -1) # Broadcost fixed weights for test
            
            x = torch.bmm(x, batch_w)
            x = torch.relu(x)
            
        # Last Layer
        batch_last_w = last_w.unsqueeze(0).expand(BATCH_QUERIES_REAL, -1, -1)
        x = torch.bmm(x, batch_last_w) # -> (10, 360k, 32)
        
        # MaxSim + Sum
        # Reshape to (10, Docs, Seq, 32)
        x = x.view(BATCH_QUERIES_REAL, BATCH_DOCS, SEQ_LEN, HEADS)
        # Max over Seq (dim 2)
        x = x.max(dim=2).values # (10, Docs, 32)
        # Sum over Heads (dim 2)
        x = x.sum(dim=2) # (10, Docs)
        
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / iters
    print(f"Average Time per Batch of {BATCH_DOCS} Docs against {BATCH_QUERIES_REAL} Queries: {avg_time:.4f} seconds")
    
    # Extrapolation
    total_docs = 8_841_823
    total_batches = total_docs / BATCH_DOCS
    
    time_per_10_queries = avg_time * total_batches
    time_per_query = time_per_10_queries / BATCH_QUERIES_REAL
    
    print(f"Estimated Time per Query: {time_per_query:.4f} seconds ({time_per_query/60:.2f} mins)")
    
if __name__ == "__main__":
    benchmark()
