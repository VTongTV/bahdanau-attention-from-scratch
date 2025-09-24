"""bidirectional encoder with annotation concatenation."""

import torch
import torch.nn as nn

from nmt.config import ExperimentConfig
from nmt.model.embedding import SourceEmbedding
from nmt.model.gru import ContextFreeCell


class Encoder(nn.Module):
    """forward and backward grus over a shared source embedding."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.embedding = SourceEmbedding(config.vocab_size, config.embedding)
        self.fwd = ContextFreeCell(config.embedding, config.hidden)
        self.bwd = ContextFreeCell(config.embedding, config.hidden)

    def forward(self, src_ids):
        """annotations (batch, seq, 2*hidden) over the source words."""
        forward, backward = self.states(src_ids)
        return torch.cat([forward, backward], dim=-1)

    def states(self, src_ids):
        """separate forward and backward state sequences."""
        batch, seq = src_ids.shape
        emb = self.embedding(src_ids)
        device = emb.device
        fwd = []
        h = torch.zeros(batch, self.fwd.hidden_size, device=device)
        for t in range(seq):
            h = self.fwd(emb[:, t], h)
            fwd.append(h)
        bwd = [None] * seq
        h = torch.zeros(batch, self.bwd.hidden_size, device=device)
        for t in reversed(range(seq)):
            h = self.bwd(emb[:, t], h)
            bwd[t] = h
        forward = torch.stack(fwd, dim=1)
        backward = torch.stack(bwd, dim=1)
        return forward, backward