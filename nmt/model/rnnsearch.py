"""rnnsearch model container wiring submodules and their init."""

import torch.nn as nn

from nmt.config import ExperimentConfig


class RNNsearch(nn.Module):
    """bidirectional gru encoder with aligned decoder. submodules are
    registered by the wiring in forward."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self._decoder_state = None

    def reset_state(self) -> None:
        """drop the decoder state between sentences."""
        self._decoder_state = None

    def init_parameters(self) -> None:
        """apply the paper init scheme over every child module."""
        from nmt.model.params import apply_paper_init

        for child in self.children():
            apply_paper_init(child)

    def decoder_state(self):
        """current decoder state or a fresh zero state."""
        if self._decoder_state is None:
            return None
        return self._decoder_state