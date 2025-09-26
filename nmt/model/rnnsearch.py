"""rnnsearch model wiring: encoder, attention, decoder, deep head."""

import torch
import torch.nn as nn

from nmt.config import ExperimentConfig
from nmt.model.attention import Attention
from nmt.model.decoder import Decoder
from nmt.model.encoder import Encoder
from nmt.model.head import DeepHead


class RNNsearch(nn.Module):
    """bidirectional encoder with aligned decoder over a maxout head."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.attention = Attention(config)
        self.decoder = Decoder(config)
        self.head = DeepHead(config)

    def forward(self, src_ids, tgt_ids, src_mask=None):
        """logits (batch, tgt_len, vocab) over the padded target."""
        annotations = self.encoder(src_ids)
        backward_first = annotations[:, 0, -self.config.hidden:]
        state = self.decoder.initial_state(backward_first)
        self.attention.cache(annotations)
        emb = self.decoder.embedding(tgt_ids)
        logits = []
        for t in range(tgt_ids.shape[1]):
            context, _ = self.attention(state, src_mask)
            logits.append(self.head.forward(state, emb[:, t], context))
            state = self.decoder.step(emb[:, t], state, context)
        return torch.stack(logits, dim=1)

    def reset_state(self) -> None:
        """drop the attention cache and decoder state between sentences."""
        self.attention._annotations = None
        self.attention.alignment._cache = None

    def init_parameters(self) -> None:
        """paper init over the whole stack (appendix b.1)."""
        self.encoder.embedding.init_parameters()
        self.encoder.fwd.init_parameters()
        self.encoder.bwd.init_parameters()
        self.decoder.init_parameters()
        self.attention.apply_initialization()
        self.head.init_parameters()