
import ir_datasets
try:
    dataset = ir_datasets.load("beir/nfcorpus")
    print("Dataset loaded.")
    count = 0
    for qrel in dataset.qrels_iter():
        count += 1
        if count >= 5: break
    print(f"Successfully read {count} qrels.")
except Exception as e:
    print(f"Error: {e}")
