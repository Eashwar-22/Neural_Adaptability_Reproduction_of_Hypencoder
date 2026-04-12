import json

# --- CONFIG ---
RUN_FILE = "logs/inference_results/retrieved_items.jsonl"
QRELS_FILE = "data/qrels.dev.jsonl"
TARGET_QID = "1030303" # "who is aziz hashim"

print(f"--- INSPECTING QUERY: {TARGET_QID} ---")

# 1. Get Ground Truth
print("\n[GROUND TRUTH] Documents marked relevant in QRELs:")
relevant_docs = []
with open(QRELS_FILE) as f:
    for line in f:
        d = json.loads(line)
        if str(d.get("query_id")) == TARGET_QID:
            relevant_docs.append(d)

if not relevant_docs:
    print(f"WARNING: Query {TARGET_QID} not found in QRELs!")
else:
    for doc in relevant_docs:
        print(f"  > DocID: {doc.get('doc_id')} (Rel: {doc.get('relevance')})")

# 2. Get Model Predictions
print("\n[MODEL RETRIEVAL] Top 10 Docs retrieved by your model:")
found = False
with open(RUN_FILE) as f:
    for line in f:
        d = json.loads(line)
        qid = str(d.get("query", {}).get("id"))
        
        if qid == TARGET_QID:
            found = True
            items = d.get("items", [])
            for i, item in enumerate(items[:10]): # Show Top 10
                doc_id = item.get("id")
                score = item.get("score")
                
                # Check if this hit is actually relevant
                is_rel = any(rd['doc_id'] == doc_id for rd in relevant_docs)
                status = "✅ MATCH" if is_rel else "❌"
                
                print(f"  #{i+1} DocID: {doc_id:<12} Score: {score:.4f}  {status}")
            break

if not found:
    print(f"Query {TARGET_QID} was not found in the run file.")