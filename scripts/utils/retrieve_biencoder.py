
import fire
import torch
from typing import Dict, List, Optional, Union
from tqdm import tqdm
from transformers import AutoTokenizer

from hypencoder_cb.inference.shared import (
    BaseRetriever,
    Item,
    TextQuery,
    load_encoded_items_from_disk,
)
from hypencoder_cb.inference.retrieve import do_retrieval_shared
from hypencoder_cb.modeling.hypencoder import TextDualEncoder
from hypencoder_cb.utils.torch_utils import dtype_lookup
from hypencoder_cb.utils.iterator_utils import batchify_slicing

class BiEncoderRetriever(BaseRetriever):

    def __init__(
        self,
        model_name_or_path: str,
        encoded_item_path: str,
        batch_size: int = 100_000,
        device: str = "cuda",
        dtype: Union[torch.dtype, str] = "float32",
        put_all_embeddings_on_device: bool = True,
        query_max_length: int = 32,
        ignore_same_id: bool = False,
    ) -> None:
        if isinstance(dtype, str):
            dtype = dtype_lookup(dtype)

        self.dtype = dtype
        self.device = device
        self.batch_size = batch_size
        self.encoded_item_path = encoded_item_path
        self.query_max_length = query_max_length
        self.ignore_same_id = ignore_same_id
        self.put_on_device = put_all_embeddings_on_device

        # Load TextDualEncoder
        print(f"Loading TextDualEncoder from {model_name_or_path}")
        self.model = (
            TextDualEncoder.from_pretrained(model_name_or_path)
            .to(device, dtype=self.dtype)
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        print("Started loading encoded items...")
        encoded_items = load_encoded_items_from_disk(
            encoded_item_path,
        )

        self.encoded_item_embeddings = torch.stack(
            [
                torch.tensor(x.representation, dtype=self.dtype)
                for x in tqdm(encoded_items)
            ]
        )

        if self.put_on_device:
            self.encoded_item_embeddings = self.encoded_item_embeddings.to(
                self.device
            )

        self.encoded_item_ids = [x.id for x in tqdm(encoded_items)]
        self.encoded_item_texts = [x.text for x in tqdm(encoded_items)]

    def retrieve(self, query: TextQuery, top_k: int) -> List[Item]:
        tokenized_query = self.tokenizer(
            query.text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.query_max_length,
        ).to(self.device)

        with torch.no_grad():
            query_output = self.model.query_encoder(
                input_ids=tokenized_query["input_ids"],
                attention_mask=tokenized_query["attention_mask"],
            )

        # Shape: (1, dim)
        query_embedding = query_output.representation

        num_batches = (
            len(self.encoded_item_embeddings) // self.batch_size
        ) + 1

        top_k_indices = torch.full((top_k * num_batches,), -1)
        top_k_scores = torch.full((top_k * num_batches,), -float("inf"))

        for batch_index, batch_item_embeddings in enumerate(
            batchify_slicing(self.encoded_item_embeddings, self.batch_size)
        ):
            if not self.put_on_device:
                batch_item_embeddings = batch_item_embeddings.to(self.device)

            # Dot product
            # query_embedding: (1, D)
            # batch_item_embeddings: (B, D)
            # similarity: (1, B) -> squeeze -> (B,)
            similarity_scores = torch.matmul(
                query_embedding, batch_item_embeddings.T
            ).squeeze(0)

            actual_k = min(top_k, similarity_scores.shape[0])
            values, indices = torch.topk(similarity_scores, actual_k, dim=0)
            
            indices = indices.cpu()
            values = values.cpu()

            start_idx = batch_index * top_k
            end_idx = start_idx + actual_k
            
            top_k_indices[start_idx:end_idx] = (
                indices + (batch_index * self.batch_size)
            )
            top_k_scores[start_idx:end_idx] = values

        final_values, indices = torch.topk(top_k_scores, top_k, dim=0)
        final_indices = top_k_indices[indices]

        items = []
        for item_idx, score in zip(final_indices, final_values):
            item_idx = item_idx.item()
            if (
                self.ignore_same_id
                and query.id == self.encoded_item_ids[item_idx]
            ):
                continue
            
            # Skip invalid indices (initialized to -1)
            if item_idx == -1:
                continue

            items.append(
                Item(
                    text=self.encoded_item_texts[item_idx],
                    id=self.encoded_item_ids[item_idx],
                    score=score.item(),
                    type="biencoder_retriever",
                )
            )

        return items


def do_retrieval(
    model_name_or_path: str,
    encoded_item_path: str,
    output_dir: str,
    ir_dataset_name: Optional[str] = None,
    query_jsonl: Optional[str] = None,
    qrel_json: Optional[str] = None,
    query_id_key: str = "id",
    query_text_key: str = "text",
    dtype: str = "fp32",
    top_k: int = 1000,
    batch_size: int = 100_000,
    query_max_length: int = 64,
    include_content: bool = True,
    do_eval: bool = True,
    metric_names: Optional[List[str]] = None,
    ignore_same_id: bool = False,
) -> None:
    
    do_retrieval_shared(
        retriever_cls=BiEncoderRetriever,
        retriever_kwargs=dict(
            model_name_or_path=model_name_or_path,
            encoded_item_path=encoded_item_path,
            dtype=dtype,
            batch_size=batch_size,
            query_max_length=query_max_length,
            ignore_same_id=ignore_same_id,
        ),
        output_dir=output_dir,
        ir_dataset_name=ir_dataset_name,
        query_jsonl=query_jsonl,
        qrel_json=qrel_json,
        query_id_key=query_id_key,
        query_text_key=query_text_key,
        top_k=top_k,
        include_content=include_content,
        do_eval=do_eval,
        metric_names=metric_names,
    )

if __name__ == "__main__":
    fire.Fire(do_retrieval)
