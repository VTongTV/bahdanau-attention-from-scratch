"""source and target embedding matrices."""

import torch
import torch.nn as nn

from nmt.model.params import gaussian, zero
from nmt.config import WEIGHT_INIT_STD


class Embedding(nn.Module):
    """word embedding matrix with a vocab guard."""

    def __init__(self, vocab_size: int, dim: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.table = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, ids):
        """lookup ids and apply dropout when enabled."""
        assert ids.max() < self.table.num_embeddings
        emb = self.table(ids)
        if self.dropout is not None:
            emb = self.dropout(emb)
        return emb

    def init_parameters(self) -> None:
        """the paper default gaussian rule for the matrix."""
        gaussian(self.table.weight, WEIGHT_INIT_STD)


class SourceEmbedding(Embedding):
    """source embedding shared by both encoder directions."""


class TargetEmbedding(Embedding):
    """target embedding. never shared with the source."""