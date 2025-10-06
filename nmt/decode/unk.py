"""unk guard: never emit <unk> during no-unk evaluation."""

import torch


def mask_unk(logits, unk_id):
    """suppress the unk logit before selection or softmax."""
    masked = logits.clone()
    masked[:, unk_id] = float("-inf")
    return masked


def drop_unk(tokens, unk_id):
    """remove unk tokens from a finished output sequence."""
    return [t for t in tokens if t != unk_id]