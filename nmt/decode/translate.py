"""low-level decode harness shared by greedy and beam search."""

import torch

from nmt.model.rnnsearch import RNNsearch


def prepare_from_annotations(model, annotations):
    """decoder start state and context closure from ready annotations."""
    if isinstance(model, RNNsearch):
        model.attention.cache(annotations)
        backward_first = annotations[:, 0, -model.config.hidden:]
    else:
        backward_first = annotations[:, -1, :model.config.hidden]
    state = model.decoder.initial_state(backward_first)
    return state, context_of(model, annotations)


def prepare_batch(model, src_ids, src_mask=None):
    """encode the source and return the decoder start state."""
    annotations = model.encoder(src_ids)
    return prepare_from_annotations(model, annotations)


def context_of(model, annotations):
    """a closure giving the context for each decoder state."""
    if isinstance(model, RNNsearch):
        def aligned(state, src_mask):
            context, _ = model.attention(state, src_mask)
            return context

        return aligned
    hidden = model.config.hidden
    zeros = torch.zeros(annotations.shape[0], hidden, device=annotations.device)
    fixed = torch.cat([annotations[:, -1, :hidden], zeros], dim=-1)
    return lambda state, src_mask: fixed