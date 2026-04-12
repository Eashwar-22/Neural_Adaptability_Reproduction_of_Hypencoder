import json

RUN_FILE = "logs/inference_results/retrieved_items.jsonl"
QRELS_FILE = "data/qrels.dev.jsonl"

print(f"--- RUN FILE SAMPLE ({RUN_FILE}) ---")
with open(RUN_FILE, 'r') as f:
    line = json.loads(f.readline())
    # print full keys to confirm structure
    print(f"Keys found: {list(line.keys())}")
    
    # Get ID format
    qid = str(line.get("query", {}).get("id"))
    
    items = line.get("items", [])
    if items:
        first_doc_id = items[0].get("id")
        print(f"Sample Query ID: '{qid}' (Type: {type(qid).__name__})")
        print(f"Sample Doc ID:   '{first_doc_id}' (Type: {type(first_doc_id).__name__})")
    else:
        print("No items found in first line.")

print("\n--- QRELS FILE SAMPLE ({QRELS_FILE}) ---")
with open(QRELS_FILE, 'r') as f:
    line = json.loads(f.readline())
    qid = str(line.get("query_id"))
    doc_id = str(line.get("doc_id"))
    
    print(f"Sample Query ID: '{qid}'")
    print(f"Sample Doc ID:   '{doc_id}'")

# Check for Intersection
print("\n--- OVERLAP CHECK ---")
run_qids = set()
with open(RUN_FILE, 'r') as f:
    for line in f:
        data = json.loads(line)
        if "query" in data:
            run_qids.add(str(data["query"]["id"]))

qrel_qids = set()
with open(QRELS_FILE, 'r') as f:
    for line in f:
        qrel_qids.add(str(json.loads(line)["query_id"]))

common = run_qids.intersection(qrel_qids)
print(f"Total Run Queries:   {len(run_qids)}")
print(f"Total QREL Queries:  {len(qrel_qids)}")
print(f"Common Query IDs:    {len(common)}")

if len(common) == 0:
    print("CRITICAL ERROR: No query IDs match. We are evaluating on the wrong query set.")