import json
import random
import os

def create_val_split(input_path, train_output_path, val_output_path, val_size=2000):
    print(f"Reading from {input_path}...")
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"Total lines: {total_lines}")
    
    if total_lines < val_size:
        raise ValueError(f"Input file is smaller than requested validation size ({val_size})")
        
    # Shuffle and split
    random.seed(42)
    random.shuffle(lines)
    
    val_lines = lines[:val_size]
    train_lines = lines[val_size:]
    
    print(f"Writing {len(val_lines)} lines to {val_output_path}...")
    with open(val_output_path, 'w') as f:
        f.writelines(val_lines)
        
    print(f"Writing {len(train_lines)} lines to {train_output_path}...")
    with open(train_output_path, 'w') as f:
        f.writelines(train_lines)
        
    print("Done!")

if __name__ == "__main__":
    base_dir = "./data"
    input_file = os.path.join(base_dir, "triples.train.jsonl")
    train_output = os.path.join(base_dir, "triples.train_split.jsonl")
    val_output = os.path.join(base_dir, "triples.val.jsonl")
    
    create_val_split(input_file, train_output, val_output)
