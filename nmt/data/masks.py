"""complementary loss masks for padded target sequences."""

import torch


def target_mask(target_ids, pad_id: int) -> torch.Tensor:
    """boolean mask over non-pad target positions."""
    return target_ids != pad_id


def loss_mask(target_ids, pad_id: int) -> torch.Tensor:
    """float mask for the nll loss. pads get zero weight."""
    return (target_ids != pad_id).float()