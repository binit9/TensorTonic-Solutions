import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Returns an embedding layer with the requested dimensions.
    """
    return nn.Embedding(vocab_size, d_model)

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Returns scaled token embeddings.
    """
    # vocab_size = torch.unique(tokens).shape[-1]
    # emb = create_embedding_layer(vocab_size, d_model)
    output = embedding(tokens)*math.sqrt(d_model)
    return output