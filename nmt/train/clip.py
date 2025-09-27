"""gradient clipping at the paper cap of 1.0."""

import torch


def clip_gradients(model, max_norm: float = 1.0):
    """shrink the global gradient norm to max_norm when it exceeds it."""
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)