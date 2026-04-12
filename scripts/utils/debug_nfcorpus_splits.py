
import ir_datasets
splits = ["beir/nfcorpus/test", "beir/nfcorpus/dev", "beir/nfcorpus/train"]

for split in splits:
    print(f"--- Checking {split} ---")
    try:
        dataset = ir_datasets.load(split)
        print(f"Provides: {dataset.has_qrels()}")
        if dataset.has_qrels():
            count = 0
            for qrel in dataset.qrels_iter():
                count += 1
                if count >= 5: break
            print(f"Successfully read {count} qrels.")
    except Exception as e:
        print(f"Error loading {split}: {e}")
