from datasets import load_dataset
import json
from tqdm import tqdm

# Download from HuggingFace
print("Downloading dataset...")
dataset = load_dataset("jfkback/hypencoder-msmarco-training-dataset", split="train")

output_file = "triples.train.jsonl"
print(f"Saving to {output_file}...")

with open(output_file, "w") as f:
    for row in tqdm(dataset):
        json.dump(row, f)
        f.write("\n")
