import ir_measures
from ir_measures import nDCG, RR, Recall
import json
import os
import sys

# --- CONFIGURATION ---
QRELS_FILE = "data/qrels.dev.jsonl"
RUN_FILE = "logs/inference_results/retrieved_items.jsonl"

# --- 1. LOAD QRELS ---
print(f"Loading QRELs from {QRELS_FILE}...")
qrels = []
if not os.path.exists(QRELS_FILE):
    print(f"ERROR: Qrels file not found at {QRELS_FILE}")
    sys.exit(1)

with open(QRELS_FILE) as f:
    for line in f:
        data = json.loads(line)
        qid = str(data.get("query_id") or data.get("id"))
        doc_id = str(data.get("doc_id") or data.get("positive_pids", [None])[0])
        rel = int(data.get("relevance", 1))
        qrels.append(ir_measures.Qrel(qid, doc_id, rel))

# --- 2. LOAD RUN ---
print(f"Loading RUN from {RUN_FILE}...")
run = []
if not os.path.exists(RUN_FILE):
    print(f"ERROR: Run file not found at {RUN_FILE}")
    sys.exit(1)

with open(RUN_FILE, 'r') as f:
    for i, line in enumerate(f):
        try:
            data = json.loads(line)
            
            # 1. Identify Query ID
            # "query": {"id": "..."} OR top-level "id"/"qid"
            if "query" in data and isinstance(data["query"], dict):
                qid = str(data["query"].get("id"))
            else:
                qid = str(data.get("query_id") or data.get("id") or data.get("qid"))
            
            # 2. Identify Hits (List of documents)
            # The error showed the key is "items"
            hits = data.get("items") or data.get("retrieved_items") or data.get("hits") or data.get("candidates")
            
            if hits:
                # Process list of candidates
                for item in hits:
                    doc_id = str(item.get("id") or item.get("doc_id") or item.get("item_id"))
                    score = float(item.get("score", 0.0))
                    run.append(ir_measures.ScoredDoc(qid, doc_id, score))
            else:
                # Fallback for flat format
                doc_id = str(data.get("doc_id") or data.get("item_id") or data.get("id"))
                score = float(data.get("score"))
                run.append(ir_measures.ScoredDoc(qid, doc_id, score))
                
        except Exception as e:
            if i == 0:
                print(f"WARNING: Issue parsing line 1: {e}")
                print(f"Line content sample: {line.strip()[:100]}...")
            continue

print(f"Loaded {len(run)} predictions.")

# --- 3. CALCULATE METRICS ---
if len(run) == 0:
    print("ERROR: No predictions loaded. Check the JSON keys in retrieved_items.jsonl")
    sys.exit(1)

print("Calculating Metrics...")
metrics = [nDCG@10, RR@10, Recall@1000]
results = ir_measures.calc_aggregate(metrics, qrels, run)

print("\n" + "="*35)
print("   FINAL SCORES (Frozen Backbone)")
print("="*35)
for metric, value in sorted(results.items(), key=lambda x: str(x[0])):
    print(f"{str(metric):<15} : {value:.4f}")
print("="*35)