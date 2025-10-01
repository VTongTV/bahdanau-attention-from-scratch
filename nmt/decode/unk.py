"""unk guard: never emit <unk> during no-unk evaluation."""

import torch


def mask_unk(logits, unk_id):
    """suppress the unk logit before selection or softmax."""
    masked = logits.clone()
    masked[:, unk_id] = float("-inf")
    return masked