
import torch
import torch.nn as nn
# CHANGED: Import the similarity function which now has the updated logic
from hypencoder_cb.modeling.similarity_colbert import no_in_batch_negatives_hypecoder_similarity

def verify_multi_head_logic():
    print("Verifying Multi-Head Hypen-ColBERT Logic...")
    
    # 1. Setup Mock Data
    num_queries = 1
    num_docs = 1
    doc_seq_len = 3
    dim = 4
    K_heads = 2 # Multi-Head with 2 heads
    
    # Document Embeddings: (Batch, SeqLen, Dim)
    # [1,0,0,0], [0,10,0,0], [0,0,1,0]
    docs = torch.tensor([
        [
            [1.0, 0.0, 0.0, 0.0], 
            [0.0, 10.0, 0.0, 0.0], 
            [0.0, 0.0, 1.0, 0.0], 
        ]
    ]).float()
    
    # 2. Mock Multi-Head Q-Net
    # We want Head 1 to focus on dim 0 (matches doc token 1)
    # We want Head 2 to focus on dim 1 (matches doc token 2)
    # Target SumMax Score = Max(Head1) + Max(Head2)
    # Head 1 on docs: [1.0, 0.0, 0.0] -> Max = 1.0
    # Head 2 on docs: [0.0, 10.0, 0.0] -> Max = 10.0
    # SumMax = 11.0
    
    class MockMultiHeadQNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_queries = 1
            # Query Vectors for 2 heads
            # Shape (2, Dim)
            self.q_vecs = torch.tensor([
                [1.0, 0.0, 0.0, 0.0], # Head 1
                [0.0, 1.0, 0.0, 0.0]  # Head 2
            ]).float()
            
        def forward(self, items):
            # items: (Batch, TotalItems, Dim)
            # We want output: (Batch, TotalItems, K)
            
            # einsum: bid, kd -> bik
            scores = torch.einsum("bid,kd->bik", items, self.q_vecs)
            return scores

    model = MockMultiHeadQNet()
    
    # 3. Test Similarity Logic
    # Input docs shape: (1, 3, 4)
    # Q-Net output shape will be (1, 3, 2)
    
    similarity = no_in_batch_negatives_hypecoder_similarity(model, docs)
    
    print(f"Computed Score: {similarity.item()}")
    
    if abs(similarity.item() - 11.0) < 1e-5:
        print("SUCCESS: Multi-Head Logic correctly implements SumMax (1.0 + 10.0 = 11.0).")
    else:
        print(f"FAILURE: Expected 11.0, got {similarity.item()}")
        
if __name__ == "__main__":
    verify_multi_head_logic()
