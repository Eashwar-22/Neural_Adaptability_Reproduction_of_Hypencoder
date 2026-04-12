import glob
import os
import re

log_files = glob.glob("logs/inference_*.log")
results = []

for log_file in log_files:
    try:
        with open(log_file, "r") as f:
            content = f.read()
            
        # Determine Model
        model = "Unknown"
        if "hypencoder_colbert" in content or "Hypen-ColBERT" in content:
            model = "HypenColBERT"
        elif "full_real_opt" in content:
            model = "Retrained Hypencoder"
        elif "hypencolbert_multihead" in content:
            model = "HypenColBERT MultiHead"
            
        # Determine Dataset
        dataset = "Unknown"
        # Look for "Dataset: <name>"
        match = re.search(r"Dataset:\s*([^\s]+)", content)
        if match:
            dataset = match.group(1)
        else:
            # Fallback to output dir
            match = re.search(r"Output Dir:.*?/([^/]+)_results", content)
            if match:
                dataset = match.group(1)

        # Check Success
        success = False
        if "[DONE]" in content or "Results saved to" in content or "saved into" in content:
             success = True
        
        # Get Timestamp from filename (approx)
        # inference_col_307233.log -> 307233 is job id, usually increasing w/ time
        job_id = 0
        match = re.search(r"_(\d+)\.log", log_file)
        if match:
            job_id = int(match.group(1))

        if model != "Unknown" and dataset != "Unknown":
            results.append({
                "file": log_file,
                "model": model,
                "dataset": dataset,
                "success": success,
                "job_id": job_id
            })
            
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")

# Group by Model + Dataset and find latest success
grouped = {}
for r in results:
    key = (r['model'], r['dataset'])
    if key not in grouped:
        grouped[key] = []
    grouped[key].append(r)

print(f"{'Model':<25} | {'Dataset':<30} | {'Status':<10} | {'File'}")
print("-" * 100)

for key, runs in grouped.items():
    # Sort by job_id descending
    runs.sort(key=lambda x: x['job_id'], reverse=True)
    
    # improved logic: find latest SUCCESS, otherwise latest FAILED
    latest_success = next((r for r in runs if r['success']), None)
    
    if latest_success:
        print(f"{key[0]:<25} | {key[1]:<30} | {'SUCCESS':<10} | {latest_success['file']}")
    else:
        latest = runs[0]
        print(f"{key[0]:<25} | {key[1]:<30} | {'FAILED':<10} | {latest['file']}")
