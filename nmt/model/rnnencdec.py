"""rnnencdec baseline: no attention, context fixed to the last forward state."""

import torch
import torch.nn as nn

from nmt.config import ExperimentConfig
from nmt.model.decoder import Decoder
from nmt.model.encoder import Encoder
from nmt.model.head import DeepHead


class RNNencdec(nn.Module):
    """cho 2014 rnn encoder-decoder built from the shared parts."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.head = DeepHead(config)

    def forward(self, src_ids, tgt_ids):
        """logits (batch, tgt_len, vocab) over the padded target."""
        batch = src_ids.shape[0]
        annotations = self.encoder(src_ids)
        forward_last = annotations[:, -1, : self.config.hidden]
        zeros = torch.zeros(batch, self.config.hidden, device=annotations.device)
        context = torch.cat([forward_last, zeros], dim=-1)
        state = self.decoder.initial_state(forward_last)
        emb = self.decoder.embedding(tgt_ids)
        logits = []
        for t in range(tgt_ids.shape[1]):
            logits.append(self.head.forward(state, emb[:, t], context))
            state = self.decoder.step(emb[:, t], state, context)
        return torch.stack(logits, dim=1)

    def init_parameters(self) -> None:
        """paper init over the whole stack."""
        self.encoder.embedding.init_parameters()
        self.encoder.fwd.init_parameters()
        self.encoder.bwd.init_parameters()
        self.decoder.init_parameters()
        self.head.init_parameters()