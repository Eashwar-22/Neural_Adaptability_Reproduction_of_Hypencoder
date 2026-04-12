import ir_datasets
import json
import random
import os
import math
import re
from collections import Counter
from tqdm import tqdm

def simple_tokenize(text):
    # Simple regex tokenizer to avoid NLTK dependency
    return re.findall(r'\b\w+\b', text.lower())

class SimpleBM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / self.corpus_size
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.corpus = corpus
        
        self._initialize()

    def _initialize(self):
        df = {}
        for document in tqdm(self.corpus, desc="Building BM25 Index"):
            self.doc_len.append(len(document))
            counts = Counter(document)
            self.doc_freqs.append(counts)
            for word in counts:
                df[word] = df.get(word, 0) + 1
        
        for word, freq in df.items():
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)

    def get_score(self, document_index, query):
        score = 0
        doc_freqs = self.doc_freqs[document_index]
        doc_len = self.doc_len[document_index]
        
        for word in query:
            if word not in doc_freqs:
                continue
            
            freq = doc_freqs[word]
            idf = self.idf.get(word, 0)
            
            numerator = idf * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += numerator / denominator
            
        return score

    def get_top_n(self, query, n=10):
        # This is a brute-force scan, but for 200k docs it might be slow in Python.
        # Ideally we'd use an inverted index, but let's try this simple version first.
        # Optimization: Only score documents that contain query terms?
        # For simplicity in this script, filtering via inverted index is better.
        
        # Build inverted index on the fly for query terms? No, pre-build it.
        # Let's start with brute force scan of indices containing at least one query term.
        
        relevant_doc_indices = set()
        for word in query:
            # We don't have an inverted index structure in this simple class for fast lookup 
            # without memory overhead. Let's just scan all docs. It might be 5-10 seconds per query.
            # With 150 queries, that's 25 minutes. Acceptable.
            pass
            
        scores = []
        for i in range(self.corpus_size):
            s = self.get_score(i, query)
            if s > 0:
                scores.append((i, s))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in scores[:n]]

def prepare_tot_data(output_path="data/triples.train_tot.jsonl"):
    print("Loading TREC TOT dataset...")
    dataset = ir_datasets.load("trec-tot/2023/train")
    
    print("Loading Documents...")
    docs = {}
    doc_tokenized = []
    doc_ids = []
    
    # Pre-load all docs
    for doc in tqdm(dataset.docs_iter(), desc="Docs"):
        docs[doc.doc_id] = doc.text
        doc_ids.append(doc.doc_id)
        doc_tokenized.append(simple_tokenize(doc.text))
        
    print(f"Loaded {len(docs)} documents.")

    # --- Hard Negative Mining Setup ---
    print("Initializing Custom BM25...")
    bm25 = SimpleBM25(doc_tokenized)
    
    # Map index to doc_id
    index_to_docid = {i: doc_id for i, doc_id in enumerate(doc_ids)}

    print("Processing Queries and Mining Negatives...")
    triples = []
    
    qrels = {}
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = []
            qrels[qrel.query_id].append(qrel.doc_id)

    # Process queries
    query_list = list(dataset.queries_iter())
    print(f"Found {len(query_list)} queries.")

    for query in tqdm(query_list, desc="Mining"):
        qid = query.query_id
        if qid not in qrels:
            continue
            
        pos_doc_ids = qrels[qid]
        query_text = query.text
        tokenized_query = simple_tokenize(query_text)
        
        # 1. Get Top Candidates (Hard Negatives)
        top_indices = bm25.get_top_n(tokenized_query, n=50)
        candidates = [index_to_docid[i] for i in top_indices]
        
        # 2. Select Hard Negatives
        hard_negatives = [doc_id for doc_id in candidates if doc_id not in pos_doc_ids]
        
        if not hard_negatives:
            # Fallback
            hard_negatives = random.sample(list(docs.keys()), k=min(10, len(docs)))

        # 3. Create Triples
        for pos_id in pos_doc_ids:
            # Depending on how many positives, maybe strict 1-to-1 or multiple triples.
            # Paper usually implies iterations. Let's make 1 triple per positive.
            neg_id = random.choice(hard_negatives)
            
            triple = {
                "query": query_text,
                "positive": docs[pos_id],
                "negatives": [docs[neg_id]] 
            }
            triples.append(triple)

    print(f"Generated {len(triples)} training triples.")
    
    # Shuffle
    random.shuffle(triples)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for t in triples:
            f.write(json.dumps(t) + "\n")
            
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    prepare_tot_data()
