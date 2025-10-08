"""alignment model: scores source steps against the decoder state."""

import torch
import torch.nn as nn

from nmt.config import ALIGNMENT_INIT_STD, ALIGNMENT_VA_INIT, ExperimentConfig
from nmt.model.params import constant, gaussian


class Alignment(nn.Module):
    """e_ij = v_a^T tanh(w_a s_{i-1} + u_a h_j)."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.w_a = nn.Linear(config.hidden, config.alignment_hidden, bias=False)
        self.u_a = nn.Linear(2 * config.hidden, config.alignment_hidden, bias=False)
        self.v_a = nn.Parameter(torch.zeros(config.alignment_hidden))
        self._cache = None

    def cache(self, annotations):
        """precompute u_a h_j once per encoder run."""
        self._cache = self.u_a(annotations)
        return self._cache

    def score(self, prev_state):
        """alignment scores for every source step. (batch, src_len)."""
        energy = self.w_a(prev_state).unsqueeze(1) + self._cache
        return torch.einsum("btk,k->bt", torch.tanh(energy), self.v_a)

    def apply_initialization(self):
        """paper init: gaussian 0.001 for w_a and u_a.
        v_a starts at 0.05, not zero. see config and methodology."""
        gaussian(self.w_a.weight, ALIGNMENT_INIT_STD)
        gaussian(self.u_a.weight, ALIGNMENT_INIT_STD)
        constant(self.v_a, ALIGNMENT_VA_INIT)