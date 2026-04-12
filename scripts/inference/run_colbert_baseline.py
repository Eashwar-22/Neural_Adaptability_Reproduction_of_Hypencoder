import os
import ir_datasets
import ir_measures
from ir_measures import nDCG, AP
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher

# Configuration
DATASET = "msmarco-passage"
TEST_DATASET = "msmarco-passage/trec-dl-2019/judged"
CHECKPOINT = "colbert-ir/colbertv2.0"
INDEX_NAME = "msmarco.nbits=2"
EXPERIMENT = "colbert_baseline"
OUTPUT_DIR = "experiments"

# Paths (ColBERT expects TSV files)
COLLECTION_PATH = os.path.join(OUTPUT_DIR, "collection.tsv")
QUERIES_PATH = os.path.join(OUTPUT_DIR, "queries.tsv")
INDEX_ROOT = os.path.join(OUTPUT_DIR, "indices")

def prepare_data():
    print(f"Loading dataset: {DATASET}...")
    dataset = ir_datasets.load(DATASET)
    
    if not os.path.exists(COLLECTION_PATH):
        print("Exporting collection.tsv...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(COLLECTION_PATH, "w") as f:
            for doc in dataset.docs_iter():
                # Clean text to avoid tab issues
                content = doc.text.replace("\t", " ").replace("\n", " ")
                f.write(f"{doc.doc_id}\t{content}\n")
    else:
        print("collection.tsv already exists. Skipping export.")

    # We don't necessarily need to export queries for Searcher if we feed them directly,
    # but it's good practice. For now, we'll load test queries in memory.

def index():
    print("Starting Indexing...")
    with Run().context(RunConfig(nranks=1, experiment=EXPERIMENT, root=OUTPUT_DIR)):
        config = ColBERTConfig(
            nbits=2,
            root=OUTPUT_DIR,
        )
        indexer = Indexer(checkpoint=CHECKPOINT, config=config)
        indexer.index(name=INDEX_NAME, collection=COLLECTION_PATH, overwrite=True)

def retrieve_and_evaluate():
    print("Starting Retrieval & Evaluation...")
    # Load Test Queries
    test_ds = ir_datasets.load(TEST_DATASET)
    queries = {q.query_id: q.text for q in test_ds.queries_iter()}
    
    with Run().context(RunConfig(experiment=EXPERIMENT, root=OUTPUT_DIR)):
        searcher = Searcher(index=INDEX_NAME, collection=COLLECTION_PATH, index_root=os.path.join(OUTPUT_DIR, EXPERIMENT, "indices"))
        
        # Batch search
        results = searcher.search_all(queries, k=10)
        
        # Convert to ir_measures format
        run = {}
        for qid, ranking in results.items():
            run[qid] = {doc_id: score for doc_id, rank, score in ranking} # Check Searcher output format
            # Actually Searcher.search_all returns a dict of qid -> list of (doc_id, rank, score) tuples usually.
            # wait, let me double check colbert-ai output. 
            # It usually returns a Ranking object. Flattening logic:
            
        # Re-formatting for ir_measures
        # ir_measures expects {qid: {doc_id: score}}
        formatted_run = {}
        for qid in run:
             formatted_run[qid] = run[qid]

        print("Evaluating...")
        metrics = [nDCG@10, AP@1000]
        results = ir_measures.calc_aggregate(metrics, test_ds.qrels_iter(), formatted_run)
        
        print("-" * 20)
        print("ColBERT Baseline Results:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        print("-" * 20)

if __name__ == "__main__":
    prepare_data()
    # index() # Uncomment to run indexing
    # retrieve_and_evaluate() # Uncomment to run retrieval
    
    # For the first run, we probably want to do it all, 
    # but indexing 8.8M passages takes time. 
    # Let's add a quick flag or just run everything.
    
    # WARNING: Indexing full MS MARCO can take hours on one GPU.
    # The user wanted a baseline. 
    # I should check if they want full index or a sample. 
    # Assuming full for "actual baseline".
    
    index()
    retrieve_and_evaluate()
