"""greedy decoding with eos stop and max length."""

import torch

from nmt.decode.translate import prepare_batch
from nmt.decode.unk import mask_unk


def greedy(model, src_ids, src_mask=None, bos_id=0, eos_id=1, unk_id=2,
           unk_suppress=False, max_len=100):
    """decode one source by always taking the top token."""
    state, context_of = prepare_batch(model, src_ids, src_mask)
    emb = model.decoder.embedding(torch.tensor([bos_id], device=src_ids.device))
    tokens = [bos_id]
    for _ in range(max_len):
        logits = model.head.forward(state, emb, context_of(state, src_mask))
        if unk_suppress and unk_id is not None:
            logits = mask_unk(logits, unk_id)
        nxt = logits.argmax(dim=-1).item()
        tokens.append(nxt)
        if nxt == eos_id:
            break
        state = model.decoder.step(emb, state, context_of(state, src_mask))
        emb = model.decoder.embedding(torch.tensor([nxt], device=src_ids.device))
    return tokens