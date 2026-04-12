"""
1. load teacher cross-encoder: BAAI/bge-reranker-v2-m3
2. read MS MARCO training triples (JSONL with 201 passages per query)
3. score each (query, passage) pair with the cross-encoder
4. write scored triples to output JSONL
"""

import json
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME  = "BAAI/bge-reranker-v2-m3"
INPUT_FILE  = "triples.train.jsonl"            # JSONL: {"query": {"content": ...}, "items": [{"content": ..., "score": ...}, ...]}
OUTPUT_FILE = "triples.scored.jsonl"
BATCH_SIZE  = 128

# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device).eval()

# scoring
with open(INPUT_FILE) as fin, open(OUTPUT_FILE, "w") as fout:
    for line in tqdm(fin, desc="Scoring"):
        record = json.loads(line)
        query = record["query"]["content"]
        items = record["items"]

        # score all (query, passage) pairs for this query
        pairs = [(query, item["content"]) for item in items]
        all_scores = []

        for i in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[i : i + BATCH_SIZE]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits.view(-1).float().cpu().tolist()
            all_scores.extend(logits)

        # write scores back
        for item, score in zip(items, all_scores):
            item["score"] = score

        fout.write(json.dumps(record) + "\n")

print("Done", OUTPUT_FILE)
