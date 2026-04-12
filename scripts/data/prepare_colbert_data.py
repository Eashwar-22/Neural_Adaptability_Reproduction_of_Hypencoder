"""
Generate ColBERTv2 teacher scores for Hypencoder distillation verification.
Uses pure transformers to load the ColBERTv2 model and compute MaxSim scores.
"""
import json
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def maxsim(query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor) -> float:
    """
    Compute MaxSim score between query and document token embeddings.
    
    MaxSim = sum over query tokens of (max similarity with any doc token)
    
    Args:
        query_embeddings: (q_len, dim) tensor
        doc_embeddings: (d_len, dim) tensor
    
    Returns:
        Scalar MaxSim score
    """
    # Normalize embeddings (ColBERTv2 uses L2 normalization)
    query_embeddings = torch.nn.functional.normalize(query_embeddings, dim=-1)
    doc_embeddings = torch.nn.functional.normalize(doc_embeddings, dim=-1)
    
    # Compute similarity matrix: (q_len, d_len)
    similarity = torch.mm(query_embeddings, doc_embeddings.T)
    
    # MaxSim: max over doc tokens for each query token, then sum
    max_sim_per_query_token = similarity.max(dim=1).values  # (q_len,)
    score = max_sim_per_query_token.sum().item()
    
    return score


def main():
    print("Loading ColBERTv2 Teacher (transformers)...")
    
    # ColBERTv2 uses bert-base-uncased as base, fine-tuned for retrieval
    # The HuggingFace version: colbert-ir/colbertv2.0
    model_name = 'colbert-ir/colbertv2.0'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)
    
    # Pre-load student tokenizer (for Hypencoder)
    student_tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')

    OUTPUT_FILE = 'data/colbert_distill_1k.jsonl'
    
    samples = []
    print("Reading triples...")
    with open('data/triples.train.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
            d = json.loads(line)
            if d['items'][0]['content'] == d['items'][1]['content']:
                continue
            samples.append(d)

    print(f"Scoring {len(samples)} samples with ColBERTv2 (MaxSim)...")
    
    with open(OUTPUT_FILE, 'w') as f_out:
        for s in tqdm(samples):
            q_text = s['query']['content']
            pos_text = s['items'][0]['content']
            neg_text = s['items'][1]['content']
            
            with torch.no_grad():
                # Tokenize query with special [Q] token handling
                # ColBERT prepends [Q] for queries and [D] for docs, but
                # the HF model handles this internally with special tokens
                q_inputs = tokenizer(
                    q_text, 
                    return_tensors='pt', 
                    truncation=True, 
                    max_length=32,
                    padding='max_length'
                ).to(device)
                
                pos_inputs = tokenizer(
                    pos_text, 
                    return_tensors='pt', 
                    truncation=True, 
                    max_length=180
                ).to(device)
                
                neg_inputs = tokenizer(
                    neg_text, 
                    return_tensors='pt', 
                    truncation=True, 
                    max_length=180
                ).to(device)
                
                # Get token embeddings (not just CLS)
                q_output = model(**q_inputs).last_hidden_state[0]  # (q_len, dim)
                pos_output = model(**pos_inputs).last_hidden_state[0]  # (pos_len, dim)
                neg_output = model(**neg_inputs).last_hidden_state[0]  # (neg_len, dim)
                
                # Compute MaxSim scores
                pos_score = maxsim(q_output, pos_output)
                neg_score = maxsim(q_output, neg_output)
            
            # Tokenize for Student (Hypencoder uses BERT tokenizer)
            q_tok = student_tokenizer(q_text, truncation=True, max_length=512)['input_ids']
            p_tok = student_tokenizer(pos_text, truncation=True, max_length=512)['input_ids']
            n_tok = student_tokenizer(neg_text, truncation=True, max_length=512)['input_ids']
            
            if p_tok == n_tok:
                continue

            record = {
                'query': {'tokenized_content': q_tok},
                'items': [
                    {'tokenized_content': p_tok, 'score': float(pos_score)},
                    {'tokenized_content': n_tok, 'score': float(neg_score)}
                ]
            }
            f_out.write(json.dumps(record) + '\n')

    print(f"Done! Saved {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
