
import torch
import torch.nn as nn
from hypencoder_cb.modeling.similarity_colbert import no_in_batch_negatives_hypecoder_similarity

def verify_hypen_colbert_logic():
    print("Verifying Hypen-ColBERT Replication Capabilities...")
    
    # 1. Setup Mock Data
    # Batch=1, 2 tokens in Query (conceptually), 3 tokens in Document, Dim=4
    # Note: Hypencoder compresses Query into Q-Net, so "2 tokens" in query 
    # just means the Q-Net *could* ideally represent 2 concepts. 
    # But Q-Net output must be scalar for the current similarity function to work as a score.
    
    num_queries = 1
    num_docs = 1
    doc_seq_len = 3
    dim = 4
    
    # Document Embeddings: (Batch, SeqLen, Dim)
    # Let's make them distinct to track which one is picked
    docs = torch.tensor([
        [
            [1.0, 0.0, 0.0, 0.0], # Token 1
            [0.0, 10.0, 0.0, 0.0], # Token 2 (High Magnitude)
            [0.0, 0.0, 1.0, 0.0], # Token 3
        ]
    ]).float() # Shape: (1, 3, 4)
    
    print(f"Document Shape: {docs.shape}")
    
    # 2. Mock Q-Net
    # We simulate a Q-Net that acts as a simple Dot Product with a "Query Vector".
    # Since Q-Net output must be scalar (Msg dim = 1), we treat it as dot product with 1 vector.
    
    class MockQNet(nn.Module):
        def __init__(self, query_vector):
            super().__init__()
            self.query_vector = query_vector # (1, 1, Dim)
            self.num_queries = 1
            
        def forward(self, items):
            # items: (Batch, NumItems*SeqLen, Dim) or (Batch, NumItems, Dim)
            # We treat items as bag of vectors
            # Score = item . query_vector
            # We need to broadcast query_vector
            
            # Check shape
            # expected input to forward is (Batch, TotalItems, Dim)
            # items is [1, 3, 4] in our case
            scores = torch.matmul(items, self.query_vector.transpose(1, 2))
            return scores
            
    # Case A: Query focuses on dim 0
    q_vec_A = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    model_A = MockQNet(q_vec_A)
    
    score_A = no_in_batch_negatives_hypecoder_similarity(model_A, docs)
    print(f"\nCase A (Query=[1,0,0,0]):")
    print(f"Doc Tokens: [1,0..], [0,10..], [0,0..]")
    print(f"Scores per token: 1.0, 0.0, 0.0")
    print(f"Hypen-ColBERT Score: {score_A.item()}")
    
    # Case B: Query focuses on dim 1
    q_vec_B = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
    model_B = MockQNet(q_vec_B)
    
    score_B = no_in_batch_negatives_hypecoder_similarity(model_B, docs)
    print(f"\nCase B (Query=[0,1,0,0]):")
    print(f"Doc Tokens: [1,0..], [0,10..], [0,0..]")
    print(f"Scores per token: 0.0, 10.0, 0.0")
    print(f"Hypen-ColBERT Score: {score_B.item()}")
    
    # 3. Can it do ColBERT?
    # ColBERT Score = Sum_over_Q_tokens( Max_over_D_tokens( q_i . d_j ) )
    # Suppose Query has 2 tokens: q1=[1,0..] and q2=[0,1..]
    # ColBERT Score = Max(q1.d) + Max(q2.d)
    #               = Max(1,0,0) + Max(0,10,0)
    #               = 1 + 10 = 11.
    
    # Hypencoder Q-Net produces ONE score per doc token.
    # S_j = QNet(d_j)
    # Final Score = Max_j(S_j)
    
    # Can we construct Q-Net such that Max_j(QNet(d_j)) == 11?
    # We need QNet(d_j) to act such that the max is 11.
    # QNet(d1) = F([1,0,0,0])
    # QNet(d2) = F([0,10,0,0])
    # QNet(d3) = F([0,0,1,0])
    
    # If Q-Net sums affinities? 
    # Say QNet(d) = (q1.d) + (q2.d). (Sum of dot products)
    # Then QNet(d1) = 1 + 0 = 1
    #      QNet(d2) = 0 + 10 = 10
    #      QNet(d3) = 0
    # Max(1, 10, 0) = 10.
    # Result 10 != 11.
    
    # Conclusion:
    # Max(Sum) != Sum(Max).
    # Hypencoder implements "Max(Sum of query terms)" (if QNet sums).
    # ColBERT implements "Sum(Max per query term)".
    
    print("\n--- CONCLUSION ---")
    print(f"Simulated ColBERT SumMax (Target): 11.0 (1.0 + 10.0)")
    print(f"Hypen-ColBERT (with Sum Q-Net): 10.0 (Max(1.0, 10.0))")
    
    if score_A.item() == 1.0 and score_B.item() == 10.0:
        print("VERIFIED: Hypen-ColBERT logic implements MaxP (Maximum Passage Token Score).")
        print("It CANNOT replicate ColBERT (Sum of MaxSims) because the aggregation order is fixed to Max(QNet(tokens)).")
    else:
        print("Something unexpected happened with the mock.")

if __name__ == "__main__":
    verify_hypen_colbert_logic()
