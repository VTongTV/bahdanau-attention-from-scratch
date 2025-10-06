"""alignment sampling: greedy decode with recorded attention weights."""

import torch

from nmt.decode.translate import prepare_batch
from nmt.decode.unk import mask_unk
from nmt.model.rnnsearch import RNNsearch


def greedy_with_align(model, src_ids, src_mask=None, bos_id=0, eos_id=1,
                      unk_id=2, unk_suppress=False, max_len=100):
    """decode one source and record the attention weights per step."""
    state, context_of = prepare_batch(model, src_ids, src_mask)
    emb = model.decoder.embedding(torch.tensor([bos_id], device=src_ids.device))
    tokens = [bos_id]
    weights = []
    for _ in range(max_len):
        context = context_of(state, src_mask)
        if isinstance(model, RNNsearch):
            weights.append(model.attention.last_weights[0].detach().cpu())
        logits = model.head.forward(state, emb, context)
        if unk_suppress and unk_id is not None:
            logits = mask_unk(logits, unk_id)
        nxt = logits.argmax(dim=-1).item()
        tokens.append(nxt)
        if nxt == eos_id:
            break
        state = model.decoder.step(emb, state, context)
        emb = model.decoder.embedding(torch.tensor([nxt], device=src_ids.device))
    aligned = torch.stack(weights) if weights else torch.zeros(0, src_ids.shape[1])
    return tokens, aligned


def sample_alignments(model, store, indices, bos_id=0, eos_id=1, unk_id=2,
                      max_len=100):
    """decode chosen rows and return tokens plus weight matrices."""
    samples = []
    for i in indices:
        src = store.src_row(i).reshape(1, -1).to(next(model.parameters()).device)
        tokens, weights = greedy_with_align(model, src, bos_id=bos_id,
                                            eos_id=eos_id, unk_id=unk_id,
                                            max_len=max_len)
        samples.append((i, src[0].tolist(), tokens, weights))
    return samples