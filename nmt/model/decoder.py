"""decoder: context gru and the initial state from the backward encoder."""

import torch
import torch.nn as nn

from nmt.config import WEIGHT_INIT_STD, ExperimentConfig
from nmt.model.embedding import TargetEmbedding
from nmt.model.gru import GRUCell
from nmt.model.params import gaussian


class DecoderCell(GRUCell):
    """gru over the target embedding, driven by a context vector."""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config.embedding, config.hidden, context_size=2 * config.hidden)

    def init_parameters(self) -> None:
        """paper init for the cell matrices."""
        GRUCell.init_parameters(self)


class Decoder(nn.Module):
    """decoder state machine. emits one state per target step."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.embedding = TargetEmbedding(config.vocab_size, config.embedding)
        self.cell = DecoderCell(config)
        self.w_s = nn.Linear(config.hidden, config.hidden, bias=False)

    def initial_state(self, backward_first):
        """s_0 = tanh(w_s * backward_h_1). the first backward state."""
        return torch.tanh(self.w_s(backward_first))

    def step(self, prev_embedding, prev_state, context):
        """one decoder step. returns the new state."""
        return self.cell(prev_embedding, prev_state, context)

    def init_parameters(self) -> None:
        """paper init: gaussian for w_s, cell init for the gru."""
        self.embedding.init_parameters()
        self.cell.init_parameters()
        gaussian(self.w_s.weight, WEIGHT_INIT_STD)