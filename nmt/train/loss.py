"""masked nll loss over the padded target."""

import torch
import torch.nn.functional as F


def token_log_probs(logits, targets):
    """log probability of the target token at every step."""
    logp = F.log_softmax(logits, dim=-1)
    return torch.gather(logp, -1, targets.unsqueeze(-1)).squeeze(-1)


def masked_nll(logits, targets, mask):
    """minibatch cost: mean nll over the sentences in the batch."""
    return sentence_nlls(logits, targets, mask).mean()


def sentence_nlls(logits, targets, mask):
    """per-sentence nll, averaged over that sentence's valid words."""
    per_token = -token_log_probs(logits, targets) * mask
    counts = mask.sum(dim=-1).clamp(min=1)
    return per_token.sum(dim=-1) / counts