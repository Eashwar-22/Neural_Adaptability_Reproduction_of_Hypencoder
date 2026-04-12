import json
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm
import os

def prepare_distillation_data(
    input_file="data/triples.train.jsonl",
    output_file="data/biencoder_distillation_10k.jsonl",
    teacher_model_name="sentence-transformers/all-MiniLM-L6-v2",
    student_tokenizer_name="google-bert/bert-base-uncased",
    max_samples=10000,
    batch_size=32
):
    print(f"Loading Teacher Model: {teacher_model_name}")
    teacher = SentenceTransformer(teacher_model_name)
    teacher.eval()
    
    print(f"Loading Student Tokenizer: {student_tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(student_tokenizer_name)
    
    print(f"Reading {input_file} (max {max_samples} samples)...")
    samples = []
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            samples.append(json.loads(line))
            
    print(f"Processing {len(samples)} samples...")
    
    with open(output_file, 'w') as f_out:
        for i in tqdm(range(0, len(samples), batch_size)):
            batch = samples[i:i+batch_size]
            
            # Extract texts
            # Structure matches hypencoder training data: 
            # {'query': {'content': ...}, 'items': [{'content': ... (pos)}, {'content': ... (neg)}, ...]}
            
            queries = [s['query']['content'] for s in batch]
            # Assumes item 0 is positive, item 1 is negative (standard for our triples)
            pos_passages = [s['items'][0]['content'] for s in batch]
            neg_passages = [s['items'][1]['content'] for s in batch]
            
            # 1. Compute Teacher Scores
            # Flatten for encoding: [q1, q2...], [p1, p2...], [n1, n2...] is not efficient for bi-encoder default
            # But SentenceTransformers encodes lists easily.
            
            q_embs = teacher.encode(queries, convert_to_tensor=True, show_progress_bar=False)
            p_embs = teacher.encode(pos_passages, convert_to_tensor=True, show_progress_bar=False)
            n_embs = teacher.encode(neg_passages, convert_to_tensor=True, show_progress_bar=False)
            
            # Compute Dot Products
            # q_embs: (B, H), p_embs: (B, H) -> (B)
            pos_scores = (q_embs * p_embs).sum(dim=1).tolist()
            neg_scores = (q_embs * n_embs).sum(dim=1).tolist()
            
            # 2. Tokenize for Student (Hypencoder)
            # The Hypencoder train.py expects:
            # {
            #   "query": {"tokenized_content": [ids...]},
            #   "items": [
            #       {"tokenized_content": [ids...], "score": 1.23},
            #       ...
            #   ]
            # }
            
            q_tokens = tokenizer(queries, add_special_tokens=True, truncation=True, max_length=512)["input_ids"]
            p_tokens = tokenizer(pos_passages, add_special_tokens=True, truncation=True, max_length=512)["input_ids"]
            n_tokens = tokenizer(neg_passages, add_special_tokens=True, truncation=True, max_length=512)["input_ids"]
            
            # 3. Write to file
            for j, sample in enumerate(batch):
                record = {
                    "query": {
                        "tokenized_content": q_tokens[j],
                        "content": queries[j],
                        "id": str(i*batch_size + j) + "_q"
                    },
                    "items": [
                        {
                            "tokenized_content": p_tokens[j],
                            "content": pos_passages[j],
                            "score": pos_scores[j],
                            "type": "pos",
                             "id": str(i*batch_size + j) + "_p"
                        },
                        {
                            "tokenized_content": n_tokens[j],
                            "content": neg_passages[j],
                            "score": neg_scores[j],
                            "type": "neg",
                             "id": str(i*batch_size + j) + "_n"
                        }
                    ]
                }
                f_out.write(json.dumps(record) + "\n")

    print(f"Saved distillation data to {output_file}")

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Check if input file exists
    if not os.path.exists("data/triples.train.jsonl"):
        print("Error: data/triples.train.jsonl not found.")
        # Fallback for verifying script if full data not present, unlikely in this user session
        # but good practice.
    else:
        prepare_distillation_data()
