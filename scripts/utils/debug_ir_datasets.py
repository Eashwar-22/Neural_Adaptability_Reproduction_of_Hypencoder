
import ir_datasets
try:
    dataset = ir_datasets.load("beir/nfcorpus/test")
    print("Dataset attributes:", dir(dataset))
    if hasattr(dataset, 'qrels_iter'):
        print("Has qrels_iter")
    else:
        print("Missing qrels_iter")
        
    # try running standard trec_eval logic
    # or just print first qrel to see structure
except Exception as e:
    print(e)
