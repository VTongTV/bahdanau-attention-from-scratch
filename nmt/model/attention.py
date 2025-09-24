"""attention: softmax weights and the weighted context vector."""

import torch
import torch.nn as nn

from nmt.config import ExperimentConfig
from nmt.model.alignment import Alignment


class Attention(nn.Module):
    """alpha_ij over source steps, c_i = sum_j alpha_ij h_j."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.alignment = Alignment(config)
        self._annotations = None

    def cache(self, annotations):
        """precompute the source encodings for this encoder run."""
        self._annotations = annotations
        self.alignment.cache(annotations)

    def forward(self, prev_state):
        """context (batch, 2n) and weights (batch, src_len)."""
        scores = self.alignment.score(prev_state)
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum("bt,btd->bd", weights, self._annotations)
        return context, weights

    def apply_initialization(self):
        self.alignment.apply_initialization()