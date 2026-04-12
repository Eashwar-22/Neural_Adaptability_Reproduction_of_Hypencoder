from datasets import load_from_disk, load_dataset
import os

dataset_path = "./data/triples.train.jsonl.dataset"
jsonl_path = "./data/triples.train.jsonl"

print(f"Checking dataset at: {dataset_path}")
try:
    ds = load_from_disk(dataset_path)
    print(f"Dataset loaded from disk.")
    print(f"Length: {len(ds)}")
    print(f"Features: {ds.features}")
except Exception as e:
    print(f"Failed to load from disk: {e}")

print("-" * 20)
# Also check if we can stream the first few lines of JSONL to confirm it's huge
print(f"Checking JSONL at: {jsonl_path}")
import subprocess
result = subprocess.run(['wc', '-l', jsonl_path], capture_output=True, text=True)
print(f"Line count: {result.stdout.strip()}")
