"""device and dtype selection."""

import torch


def pick_device(preference: str = "auto") -> torch.device:
    """return the best available device for the preference."""
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preference == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(name: str = "float32") -> torch.dtype:
    """return the torch dtype for the name."""
    return getattr(torch, name)