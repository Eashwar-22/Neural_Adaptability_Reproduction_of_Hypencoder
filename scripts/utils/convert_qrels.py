import json
import collections

input_path = "data/qrels.dev.jsonl"
output_path = "data/qrels.dev.json"

qrels = collections.defaultdict(dict)

print(f"Reading from {input_path}...")
with open(input_path, "r") as f:
    for line in f:
        item = json.loads(line)
        # item: {"query_id": "...", "doc_id": "...", "relevance": ...}
        qid = str(item["query_id"])
        did = str(item["doc_id"])
        rel = int(item["relevance"])
        qrels[qid][did] = rel

print(f"Loaded {len(qrels)} queries.")
print(f"Writing to {output_path}...")

with open(output_path, "w") as f:
    json.dump(qrels, f, indent=4)

print("Done.")
