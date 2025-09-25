"""deep output head: maxout pooling and the vocabulary projection."""

import torch
import torch.nn as nn

from nmt.config import WEIGHT_INIT_STD, ExperimentConfig
from nmt.model.params import gaussian, zero


class MaxoutHead(nn.Module):
    """u_o s + v_o e + c_o c stacked to 2l, pooled in pairs to l."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.u_o = nn.Linear(config.hidden, 2 * config.maxout, bias=False)
        self.v_o = nn.Linear(config.embedding, 2 * config.maxout, bias=False)
        self.c_o = nn.Linear(2 * config.hidden, 2 * config.maxout, bias=False)
        self.b = nn.Parameter(torch.zeros(2 * config.maxout))

    def forward(self, state, embedding, context):
        """pooled units t_i of size l. (batch, maxout)."""
        pre = self.u_o(state) + self.v_o(embedding) + self.c_o(context) + self.b
        return torch.max(pre.unflatten(-1, (-1, 2)), dim=-1).values

    def init_parameters(self) -> None:
        """paper init: gaussian weights, zero bias."""
        for lin in (self.u_o, self.v_o, self.c_o):
            gaussian(lin.weight, WEIGHT_INIT_STD)
        zero(self.b)