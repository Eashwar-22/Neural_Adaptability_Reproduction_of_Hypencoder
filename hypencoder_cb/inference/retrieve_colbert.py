from pathlib import Path
from typing import Dict, List, Optional, Union, Generator, Tuple
import heapq
import fire
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from hypencoder_cb.inference.shared import (
    BaseRetriever,
    Item,
    TextQuery,
    retrieve_for_ir_dataset_queries,
    retrieve_for_jsonl_queries,
    items_from_ir_dataset,
    items_from_jsonl,
)
from hypencoder_cb.modeling.hypencoder_colbert import HypencoderDualEncoder
from hypencoder_cb.utils.data_utils import (
    load_qrels_from_ir_datasets,
    load_qrels_from_json,
)
from hypencoder_cb.utils.eval_utils import (
    calculate_metrics_to_file,
    load_standard_format_as_run,
    pretty_print_standard_format,
    do_eval_and_pretty_print,
)
from hypencoder_cb.utils.torch_utils import dtype_lookup
from hypencoder_cb.modeling.similarity_colbert import no_in_batch_negatives_hypecoder_similarity

def batch_iterator(iterable, batch_size):
    """Yields batches from an iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

class HypencoderColBERTShardedRetriever(BaseRetriever):
    def __init__(
        self,
        model_name_or_path: str,
        document_source: str, # "ir_datasets:name" or "jsonl:path"
        doc_batch_size: int = 2000, # Number of *documents* to load/encode at once (RAM limit)
        query_batch_size: int = 100, # Number of queries to process in parallel against a doc chunk
        device: str = "cuda",
        dtype: Union[torch.dtype, str] = "bfloat16",
        query_max_length: int = 32,
        doc_max_length: int = 180, # ColBERT standard
        ignore_same_id: bool = False,
        aggregation_strategy: str = "max",
    ) -> None:
        if isinstance(dtype, str):
            dtype = dtype_lookup(dtype)

        self.dtype = dtype
        self.device = device
        self.doc_batch_size = doc_batch_size
        self.query_batch_size = query_batch_size
        self.document_source = document_source
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length
        self.ignore_same_id = ignore_same_id
        self.aggregation_strategy = aggregation_strategy

        print(f"Loading HypencoderColBERT model from {model_name_or_path}...")
        self.model = (
            HypencoderDualEncoder.from_pretrained(model_name_or_path)
            .to(device, dtype=self.dtype)
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        print("Model loaded.")

    def _get_document_iterator(self) -> Generator[List[Item], None, None]:
        """Returns an iterator over batches of documents."""
        if self.document_source.startswith("ir_datasets:"):
            ds_name = self.document_source.split(":", 1)[1]
            items_iter = items_from_ir_dataset(ds_name)
        elif self.document_source.startswith("jsonl:"):
            path = self.document_source.split(":", 1)[1]
            items_iter = items_from_jsonl(path)
        else:
            raise ValueError(f"Unknown document source format: {self.document_source}")
        
        return batch_iterator(items_iter, self.doc_batch_size)

    def retrieve_all(self, queries: List[TextQuery], top_k: int) -> List[Tuple[TextQuery, List[Item]]]:
        """
        Specialized retrieval method that processes ALL queries against the streamed document collection.
        This overrides the standard per-query `retrieve` method to allow for efficient scanning.
        """
        
        # 1. Initialize heaps for each query
        # Structure: query_id -> list of (-score, doc_id, doc_text) for min-heap behavior
        # We store negative score because python heapq is a min-heap, and we want to keep the top scores.
        # Actually, python's heapq.nlargest is easier, but for streaming we usually maintain a heap of size K.
        # Let's use a simple list and sort/truncate occasionally or at the end if K is small.
        # For efficiency with large K, a heap is better.
        
        query_heaps = {q.id: [] for q in queries} 
        
        # 2. Pre-encode queries (weights/parameters)
        # Hypencoder queries produce *Parameters* for the Q-Net.
        # We can pre-compute these parameters for all queries.
        
        print("Encoding queries...")
        all_query_models = []
        
        # We batch query encoding to avoid OOM if many queries
        for query_batch in tqdm(batch_iterator(queries, self.query_batch_size), desc="Encoding Queries"):
             batch_texts = [q.text for q in query_batch]
             tokenized_queries = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.query_max_length,
            ).to(self.device)
             
             with torch.no_grad():
                # Get the Q-Net parameters (Hypernetwork output)
                query_output = self.model.query_encoder(
                    input_ids=tokenized_queries["input_ids"],
                    attention_mask=tokenized_queries["attention_mask"],
                )
                # This is the "Model" (Q-Net) for each query
                all_query_models.append(query_output.representation)

        # 3. Stream Documents
        print(f"Streaming documents from {self.document_source}...")
        
        doc_count = 0
        
        # We need to iterate through queries in batches during the scan to manage memory
        # But we already have the `query_models`.
        # The `query_models` object in Hypencoder usually holds parameters for a batch.
        # We can't easily concatenate them if they are arbitrary python objects (NoTorchSequential).
        # However, `HypencoderOutput.representation` is a `NoTorchSequential` containing tensors.
        # We can iterate over the list of `all_query_models` (which are batches) and apply them to doc chunks.
        
        for doc_batch in tqdm(self._get_document_iterator(), desc="Scanning Docs"):
            doc_texts = [d.text for d in doc_batch]
            doc_ids = [d.id for d in doc_batch]
            
            # Encode Documents on GPU
            tokenized_docs = self.tokenizer(
                doc_texts,
                return_tensors="pt",
                padding="longest", # Pad to longest in this specific batch (efficient)
                truncation=True,
                max_length=self.doc_max_length,
            ).to(self.device)
            
            with torch.no_grad():
                # TextEncoder -> (Batch, SeqLen, Hidden)
                # With pooling="none" + projection -> (2000, L, 128)
                doc_output = self.model.passage_encoder(
                    input_ids=tokenized_docs["input_ids"],
                    attention_mask=tokenized_docs["attention_mask"],
                )
                doc_embeddings = doc_output.representation
                
                # Check 3D shape safety (Hypen-ColBERT is 3D)
                if len(doc_embeddings.shape) == 2:
                    # Fallback for 2D models if needed, though this class is for ColBERT
                    doc_embeddings = doc_embeddings.unsqueeze(1)
            
            # Now score this chunk against ALL query batches
            query_offset = 0
            for query_model_batch in all_query_models:
                # query_model_batch is a NoTorchSequential for N queries
                num_queries_in_batch = query_model_batch.num_queries
                
                # Compute Similarity: (NumQueries, NumDocs)
                # doc_embeddings: (NumDocs, SeqLen, Dim)
                # query_model_batch expects input: (NumQueries, NumItems, Dim)
                # But here we have Many Queries vs Many Docs.
                # Hypencoder's standard `no_in_batch_negatives` assumes 1-to-1 or 1-to-N structure where N is fixed.
                # We need to broadcast.
                
                # To score N queries against M docs:
                # We can cheat: Treating Docs as "Items" for a single "Query" doesn't work directly because param shapes differ.
                # Correct approach for Hypencoder:
                # The Q-Net parameters are specific to each query q_i.
                # We want q_i(d_j).
                # `query_model_batch` holds params of shape (NumQueries, ...).
                # `doc_embeddings` is (NumDocs, SeqLen, Dim).
                
                # We need to reshape doc_embeddings to (1, NumDocs*SeqLen, Dim) and broadcast?
                # No, the `NoTorchLinear` does `einsum("qin,qnh->qih", x, weight)`.
                # x is (NumQueries, NumItems, InDim).
                # we want x to represent the SAME docs for all queries.
                # So x should be (NumQueries, NumDocs*SeqLen, InDim).
                
                M, L, D = doc_embeddings.shape
                
                # Expand docs for the query batch
                # (NumDocs, L, D) -> (1, NumDocs*L, D) -> (NumQueries, NumDocs*L, D)
                # This could be memory heavy.
                # 100 queries * 2000 docs * 150 tokens * 128 floats * 2 bytes = 7.6 GB.
                # This fits in H100 (80GB). If not, reduce doc_batch_size or query_batch_size.
                
                doc_embeddings_flat = doc_embeddings.view(1, M*L, D).expand(num_queries_in_batch, -1, -1)
                
                # Forward Pass through Q-Net
                # Returns (NumQueries, NumDocs*L, NumHeads) usually
                scores_flat = query_model_batch(doc_embeddings_flat)
                
                # Handle Multi-Head Output
                if len(scores_flat.shape) == 3 and scores_flat.shape[-1] > 1:
                    num_heads = scores_flat.shape[-1]
                    # Reshape: (NumQueries, NumDocs*L, NumHeads) -> (NumQueries, NumDocs, L, NumHeads)
                    scores_4d = scores_flat.view(num_queries_in_batch, M, L, num_heads)
                    
                    if self.aggregation_strategy == "sum_max":
                         # Multi-Head ColBERT: Sum of MaxSims per head
                         # Max over L (SeqLen) -> (Q, D, H)
                         # Sum over H (Heads) -> (Q, D)
                         scores = scores_4d.max(dim=2).values.sum(dim=-1)
                    else:
                         # Default fallback: Sum-Max is robust
                         scores = scores_4d.max(dim=2).values.sum(dim=-1)
                else:
                    # Single-Head Logic
                    # Flatten heads if it's (..., 1) -> (...)
                    if len(scores_flat.shape) == 3:
                        scores_flat = scores_flat.squeeze(-1)
                        
                    # Reshape back to (NumQueries, NumDocs, SeqLen)
                    scores_3d = scores_flat.view(num_queries_in_batch, M, L)
                    
                    # Aggregation
                    if self.aggregation_strategy == "sum_top_5":
                        # Sum of Top-5 activations (Evidence Accumulation)
                        k = min(5, L)
                        scores = scores_3d.topk(k, dim=-1).values.sum(dim=-1)
                    else:
                        # MaxSim: Max over sequence dimension (Default)
                        scores = scores_3d.max(dim=-1).values
                
                # Update Heaps
                scores_cpu = scores.float().cpu().numpy()
                
                current_query_ids = [q.id for q in queries[query_offset : query_offset + num_queries_in_batch]]
                
                for i, q_id in enumerate(current_query_ids):
                    q_scores = scores_cpu[i] # (NumDocs,)
                    for doc_idx, score in enumerate(q_scores):
                         # If heap is not full, push
                         # If heap is full, pushpop if score is higher
                         if len(query_heaps[q_id]) < top_k:
                             heapq.heappush(query_heaps[q_id], (score, doc_ids[doc_idx], doc_texts[doc_idx]))
                         else:
                             if score > query_heaps[q_id][0][0]:
                                 heapq.heapreplace(query_heaps[q_id], (score, doc_ids[doc_idx], doc_texts[doc_idx]))
                
                query_offset += num_queries_in_batch
            
            doc_count += len(doc_batch)

        return self._format_results(query_heaps)

    def _format_results(self, query_heaps: Dict) -> List[Tuple[TextQuery, List[Item]]]:
        results = []
        for q_id, heap in query_heaps.items():
            # Sort by score descending
            sorted_items = sorted(heap, key=lambda x: x[0], reverse=True)
            
            item_objs = [
                Item(id=doc_id, text=doc_text, score=float(score))
                for score, doc_id, doc_text in sorted_items
            ]
            results.append((TextQuery(id=q_id), item_objs))
        return results

    # Stub required by abstract base class, but we use retrieve_all
    def retrieve(self, query: TextQuery, top_k: int) -> List[Item]:
        raise NotImplementedError("Use retrieve_all for sharded retrieval.")


def do_retrieval(
    model_name_or_path: str,
    output_dir: str,
    document_source: str, # e.g. "ir_datasets:msmarco-passage/dev/small"
    ir_dataset_name: Optional[str] = None, # For queries/qrels
    query_jsonl: Optional[str] = None, # Alternative for queries
    qrel_json: Optional[str] = None,
    dtype: str = "bfloat16",
    top_k: int = 1000,
    doc_batch_size: int = 2000,
    query_batch_size: int = 100,
    query_max_length: int = 32,
    doc_max_length: int = 180,
    do_eval: bool = True,
    metric_names: Optional[List[str]] = None,
    aggregation_strategy: str = "max",
) -> None:
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_file = output_dir / "retrieved_items.jsonl"
    metric_dir = output_dir / "metrics"

    retriever = HypencoderColBERTShardedRetriever(
        model_name_or_path=model_name_or_path,
        document_source=document_source,
        doc_batch_size=doc_batch_size,
        query_batch_size=query_batch_size,
        dtype=dtype,
        query_max_length=query_max_length,
        doc_max_length=doc_max_length,
        aggregation_strategy=aggregation_strategy,
    )

    # Load Queries
    queries = []
    if ir_dataset_name:
        import ir_datasets
        ds = ir_datasets.load(ir_dataset_name)
        queries = [TextQuery(id=q.query_id, text=q.text) for q in ds.queries_iter()]
    elif query_jsonl:
         # Simplified loading
         for line in items_from_jsonl(query_jsonl, text_key="query_text", id_key="query_id"): # Assuming generic Item structure
             queries.append(TextQuery(id=line.id, text=line.text))
    else:
        raise ValueError("Must provide ir_dataset_name or query_jsonl for queries.")

    print(f"Loaded {len(queries)} queries.")
    
    # Run Retrieval
    results = retriever.retrieve_all(queries, top_k)
    
    # Save Results using shared utils
    from hypencoder_cb.inference.shared import query_items_to_jsonl
    query_items_to_jsonl(results, str(retrieval_file))
    
    # Eval
    if do_eval:
        do_eval_and_pretty_print(
            ir_dataset_name=ir_dataset_name,
            qrel_json=qrel_json,
            retrieval_path=str(retrieval_file),
            output_dir=str(metric_dir),
            metric_names=metric_names,
        )

if __name__ == "__main__":
    fire.Fire(do_retrieval)
