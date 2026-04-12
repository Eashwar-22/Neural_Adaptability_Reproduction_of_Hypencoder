
import ir_datasets
try:
    dataset = ir_datasets.load("beir/nfcorpus")
    print(f"Dataset: {dataset}")
    print(f"Has qrels_iter? {hasattr(dataset, 'qrels_iter')}")
    print(f"Has qrels_defs? {hasattr(dataset, 'qrels_defs')}")
    print(f"Dir: {dir(dataset)}")
    
    # Try accessing qrels via other sub-datasets if needed
    # sometimes main dataset is just a container?
except Exception as e:
    print(f"Error: {e}")
