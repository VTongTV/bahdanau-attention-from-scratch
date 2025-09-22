"""parameter initialization per the paper appendix b.1."""

import torch


def orthogonal(tensor) -> None:
    """init a recurrent weight matrix as random orthogonal."""
    torch.nn.init.orthogonal_(tensor)