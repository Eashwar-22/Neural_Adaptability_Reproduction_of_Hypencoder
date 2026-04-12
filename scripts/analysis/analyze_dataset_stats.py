
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

def analyze_stats(data_path="data/triples.train.jsonl", max_lines=100000):
    print(f"Analyzing up to {max_lines} lines from {data_path}...")
    
    query_lengths = []
    pos_passage_lengths = []
    neg_passage_lengths = []
    total_samples = 0
    unique_queries = set()
    
    with open(data_path, 'r') as f:
        for i, line in tqdm(enumerate(f), total=max_lines):
            if i >= max_lines:
                break
            try:
                data = json.loads(line)
                # Structure: {"query": {"content":...}, "items": [{"content":...}, ...]}
                
                q = data.get("query", {}).get("content", "")
                if q:
                    query_lengths.append(len(q.split()))
                    unique_queries.add(q)
                
                items = data.get("items", [])
                # Usually first is positive, rest are negatives in standard implementations, 
                # but let's check score or assumptions. 
                # In train.py: positive_filter_type="first" -> items[:1] is positive.
                
                if len(items) > 0:
                    p_content = items[0].get("content", "")
                    pos_passage_lengths.append(len(p_content.split()))
                    
                for item in items[1:]:
                    n_content = item.get("content", "")
                    neg_passage_lengths.append(len(n_content.split()))
                    
                total_samples += 1
            except:
                continue

    print(f"Total Samples Analyzed: {total_samples}")
    print(f"Unique Queries: {len(unique_queries)}")
    print(f"Avg Query Length: {np.mean(query_lengths):.2f} words")
    print(f"Avg Pos Passage Length: {np.mean(pos_passage_lengths):.2f} words")
    print(f"Avg Neg Passage Length: {np.mean(neg_passage_lengths):.2f} words")
    
    # Save Report
    output_dir = "docs/analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {
        "total_samples_analyzed": total_samples,
        "unique_queries": len(unique_queries),
        "query_len_mean": float(np.mean(query_lengths)),
        "query_len_std": float(np.std(query_lengths)),
        "pos_len_mean": float(np.mean(pos_passage_lengths)),
        "neg_len_mean": float(np.mean(neg_passage_lengths))
    }
    
    with open(f"{output_dir}/msmarco_dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
        
    return stats

if __name__ == "__main__":
    analyze_stats()
