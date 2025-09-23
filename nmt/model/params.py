"""parameter initialization per the paper appendix b.1."""

import torch

from nmt.config import WEIGHT_INIT_STD


def orthogonal(tensor) -> None:
    """init a recurrent weight matrix as random orthogonal."""
    torch.nn.init.orthogonal_(tensor)


def gaussian(tensor, std) -> None:
    """init a weight matrix from a zero-mean gaussian."""
    torch.nn.init.normal_(tensor, mean=0.0, std=std)


def zero(tensor) -> None:
    """zero a vector such as a bias or the alignment vector."""
    torch.nn.init.zeros_(tensor)


def default(param) -> None:
    """the paper default: gaussian 0.01 weights, zero vectors."""
    if param.dim() > 1:
        gaussian(param, WEIGHT_INIT_STD)
    else:
        zero(param)


def apply_paper_init(module) -> None:
    """init every parameter of a module tree per appendix b.1.

    a module with its own init_parameters takes over its subtree.
    anything else falls to the default rule.
    """

    def walk(m):
        init_parameters = getattr(m, "init_parameters", None)
        if init_parameters is not None:
            init_parameters()
            return
        for p in m.parameters(recurse=False):
            default(p)
        for child in m.children():
            walk(child)

    walk(module)