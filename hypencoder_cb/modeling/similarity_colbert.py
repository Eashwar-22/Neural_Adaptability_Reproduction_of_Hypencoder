from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

from hypencoder_cb.modeling.shared import EncoderOutput
from hypencoder_cb.modeling.similarity_and_losses import (
    SimilarityAndLossBase,
    SimilarityAndLossOutput,
    MarginMSELoss,
    CrossEntropyLoss,
    pos_neg_triplets_from_similarity,
)

def no_in_batch_negatives_hypecoder_similarity(
    query_models: Callable,
    item_embeddings: torch.Tensor,
    required_num_items_per_query: Optional[int] = None,
) -> torch.Tensor:
    """Same signature as original, but supports 3D item_embeddings for ColBERT-like MaxSim."""
    
    # CASE 1: Standard Hypencoder (2D: N x Hidden)
    if len(item_embeddings.shape) == 2:
        num_items, item_emb_dim = item_embeddings.shape
        num_queries = query_models.num_queries
        
        assert num_items % num_queries == 0
        num_items_per_query = num_items // num_queries

        if required_num_items_per_query is not None:
            assert num_items_per_query == required_num_items_per_query

        # (Num Queries, Num Items Per Query, Hidden)
        item_embeddings = item_embeddings.view(
            num_queries, num_items_per_query, item_emb_dim
        )
        
        # Score -> Squeeze to (Num Queries, Num Items Per Query)
        similarity = query_models(item_embeddings).squeeze()
        return similarity

    # CASE 2: Hypen-ColBERT (3D: N x SeqLen x Hidden)
    elif len(item_embeddings.shape) == 3:
        num_items, seq_len, item_emb_dim = item_embeddings.shape
        num_queries = query_models.num_queries
        
        assert num_items % num_queries == 0
        num_items_per_query = num_items // num_queries

        if required_num_items_per_query is not None:
            assert num_items_per_query == required_num_items_per_query
            
        # Reshape to flatten SeqLen into ItemsPerQuery for Q-Net processing
        # Target: (Num Queries, Num Items * SeqLen, Hidden)
        # Note: We effectively treat every token as an "item" for the Q-Net
        item_embeddings_flat = item_embeddings.view(
            num_queries, 
            num_items_per_query * seq_len, 
            item_emb_dim
        )
        
        # Score: (Num Queries, Num Items * SeqLen, K)
        scores_flat = query_models(item_embeddings_flat)
        
        # Check output dimension to decide aggregation strategy
        output_dim = scores_flat.shape[-1]
        
        if output_dim == 1:
            # CASE 2a: Standard Hypen-ColBERT (1 Head -> MaxP)
            # Reshape back: (Num Queries, Num Items, SeqLen)
            scores_3d = scores_flat.view(
                num_queries, num_items_per_query, seq_len
            )
            # MaxSim Aggregation: Max over SeqLen dimension
            similarity = scores_3d.max(dim=-1).values
        else:
            # CASE 2b: Multi-Head Hypen-ColBERT (K Heads -> SumMax)
            # Shape: (Num Queries, Num Items, SeqLen, K)
            scores_4d = scores_flat.view(
                num_queries, num_items_per_query, seq_len, output_dim
            )
            
            # 1. Max over Doc Tokens (SeqLen dimension 2)
            # Result: (Num Queries, Num Items, K)
            max_scores_per_head = scores_4d.max(dim=2).values
            
            # 2. Sum over Heads (Dimension 2)
            # Result: (Num Queries, Num Items)
            similarity = max_scores_per_head.sum(dim=-1)
        
        return similarity
    
    else:
        raise ValueError(f"Input shape {item_embeddings.shape} not supported. Expected 2D or 3D.")


def in_batch_negatives_hypecoder_similarity(
    query_models: Callable,
    item_embeddings: torch.Tensor,
    required_num_items_per_query: Optional[int] = None,
) -> torch.Tensor:
    # CASE 1: Standard Hypencoder (2D: N x Hidden)
    if len(item_embeddings.shape) == 2:
        num_items, item_emb_dim = item_embeddings.shape
        num_queries = query_models.num_queries

        # Expand items for each query: (Num Queries, Num Items, Hidden)
        item_embeddings = item_embeddings.unsqueeze(0).repeat(num_queries, 1, 1)
        
        # Run Q-Net
        # Output: (Num Queries, Num Items, 1) -> Squeeze to (Num Queries, Num Items)
        similarity = (
            query_models(item_embeddings).view(num_queries, num_items).squeeze()
        )
        return similarity

    # CASE 2: Hypen-ColBERT (3D: N x SeqLen x Hidden)
    elif len(item_embeddings.shape) == 3:
        num_items, seq_len, item_emb_dim = item_embeddings.shape
        num_queries = query_models.num_queries
        
        # We need to score EVERY item against EVERY query.
        # Target Input to Q-Net: (Num Queries, Num Items * SeqLen, Hidden)
        
        # Expand items for each query: (Num Queries, Num Items, SeqLen, Hidden)
        item_embeddings_expanded = item_embeddings.unsqueeze(0).expand(num_queries, -1, -1, -1)
        
        # Flatten Num Items and SeqLen: (Num Queries, Num Items * SeqLen, Hidden)
        item_embeddings_flat = item_embeddings_expanded.reshape(
            num_queries, num_items * seq_len, item_emb_dim
        )
        
        # Run Q-Net
        # Output: (Num Queries, Num Items * SeqLen, K)
        scores_flat = query_models(item_embeddings_flat)
        output_dim = scores_flat.shape[-1]
        
        # Reshape back: (Num Queries, Num Items, SeqLen, K)
        scores_4d = scores_flat.view(
            num_queries, num_items, seq_len, output_dim
        )
        
        # Aggregation Logic (Same as no_in_batch)
        if output_dim == 1:
             # Max over SeqLen: (Num Queries, Num Items, 1)
             max_scores = scores_4d.max(dim=2).values
             # Squeeze last dim: (Num Queries, Num Items)
             similarity = max_scores.squeeze(-1)
        else:
             # Multi-Head: SumMax
             # 1. Max over SeqLen (dim 2): (Num Queries, Num Items, K)
             max_scores_per_head = scores_4d.max(dim=2).values
             # 2. Sum over Heads (dim 2): (Num Queries, Num Items)
             similarity = max_scores_per_head.sum(dim=-1)
             
        return similarity

    else:
        raise ValueError(f"Input shape {item_embeddings.shape} not supported. Expected 2D or 3D.")


class HypencoderMarginMSELoss(MarginMSELoss):
    def _get_similarity(
        self,
        query_output: EncoderOutput,
        passage_output: EncoderOutput,
        **kwargs,
    ) -> torch.Tensor:
        # Use our NEW similarity function
        similarity = no_in_batch_negatives_hypecoder_similarity(
            query_output.representation,
            passage_output.representation,
        )

        return pos_neg_triplets_from_similarity(similarity)

    def forward(
        self,
        query_output: EncoderOutput,
        passage_output: EncoderOutput,
        labels: Optional[torch.Tensor] = None,
    ) -> SimilarityAndLossOutput:
        loss = torch.tensor(0.0, device=passage_output.representation.device)
        similarity = self._get_similarity(query_output, passage_output)
        
        loss += self.scale * self._loss(
            similarity,
            labels,
        )

        return SimilarityAndLossOutput(similarity=similarity, loss=loss)


class HypencoderCrossEntropyLoss(CrossEntropyLoss):
    def __init__(
        self,
        use_query_embedding_representation: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_query_embedding_representation = (
            use_query_embedding_representation
        )

    def _get_similarity(
        self,
        query_output: EncoderOutput,
        passage_output: EncoderOutput,
        **kwargs,
    ) -> torch.Tensor:
        if self.use_in_batch_negatives:
            if self.use_cross_device_negatives:
                raise NotImplementedError(
                    "Cross device negatives not supported for Hypencoder."
                )
            else:
                query_model = query_output.representation
                passage_embeddings = passage_output.representation

            if self.only_use_first_item:
                num_items = passage_embeddings.shape[0]
                num_queries = query_model.num_queries
                items_per_query = num_items // num_queries

                indices = (
                    torch.arange(
                        num_queries,
                        device=passage_embeddings.device,
                        dtype=torch.long,
                    )
                    * items_per_query
                )

                passage_embeddings = passage_embeddings[indices]

            similarity = in_batch_negatives_hypecoder_similarity(
                query_model, passage_embeddings
            )
        else:
            # Use our NEW similarity function
            similarity = no_in_batch_negatives_hypecoder_similarity(
                query_output.representation, passage_output.representation
            )

        return similarity
