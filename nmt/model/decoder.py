"""decoder: context gru and the initial state from the backward encoder."""

import torch.nn as nn

from nmt.config import ExperimentConfig
from nmt.model.gru import GRUCell


class DecoderCell(GRUCell):
    """gru over the target embedding, driven by a context vector."""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config.embedding, config.hidden, context_size=2 * config.hidden)

    def init_parameters(self) -> None:
        """paper init for the cell matrices."""
        GRUCell.init_parameters(self)