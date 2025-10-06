"""beam search decoding with width and log-prob accumulation."""

import torch

from nmt.decode.translate import prepare_batch
from nmt.decode.unk import mask_unk


def beam_search(model, src_ids, src_mask=None, bos_id=0, eos_id=1, unk_id=2,
                beam_size=10, unk_suppress=False, max_len=100, trace=None):
    """decode one source keeping the top beam_size paths."""
    state, context_of = prepare_batch(model, src_ids, src_mask)
    emb = model.decoder.embedding(torch.tensor([bos_id], device=src_ids.device))
    beams = [(0.0, [bos_id], state, emb)]
    finished = []
    for step in range(max_len):
        if not beams:
            break
        candidates = []
        for score, tokens, beam_state, beam_emb in beams:
            context = context_of(beam_state, src_mask)
            logits = model.head.forward(beam_state, beam_emb, context)
            if unk_suppress and unk_id is not None:
                logits = mask_unk(logits, unk_id)
            logps = torch.log_softmax(logits, dim=-1)
            top = logps.topk(beam_size)
            for value, index in zip(top.values.squeeze(0), top.indices.squeeze(0)):
                token = index.item()
                new_score = score + value.item()
                if token == eos_id:
                    finished.append((new_score, tokens + [token], beam_state, beam_emb))
                else:
                    next_emb = model.decoder.embedding(
                        torch.tensor([token], device=src_ids.device))
                    next_state = model.decoder.step(next_emb, beam_state, context)
                    candidates.append((new_score, tokens + [token], next_state, next_emb))
        beams = sorted(candidates, key=lambda b: b[0], reverse=True)[:beam_size]
        if trace is not None:
            trace(step, [(score, tokens) for score, tokens, _, _ in beams])
    all_beams = sorted(beams + finished, key=lambda b: b[0], reverse=True)
    return [tokens for _, tokens, _, _ in all_beams[:beam_size]]