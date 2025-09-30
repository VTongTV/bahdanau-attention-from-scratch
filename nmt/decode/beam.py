"""beam search decoding with width and log-prob accumulation."""

import torch

from nmt.decode.translate import prepare_batch


def beam_search(model, src_ids, src_mask=None, bos_id=1, eos_id=2, unk_id=0,
                beam_size=10, unk_suppress=False, max_len=100):
    """decode one source keeping the top beam_size paths."""
    state, context_of = prepare_batch(model, src_ids, src_mask)
    emb = model.decoder.embedding(torch.tensor([bos_id], device=src_ids.device))
    beams = [(0.0, [bos_id], state, emb)]
    finished = []
    for _ in range(max_len):
        if not beams:
            break
        candidates = []
        next_state = None
        for score, tokens, beam_state, beam_emb in beams:
            context = context_of(beam_state, src_mask)
            logps = model.head.log_probs(beam_state, beam_emb, context)
            if unk_suppress and unk_id is not None:
                logps = logps.clone()
                logps[:, unk_id] = float("-inf")
            values, indices = logps.topk(beam_size).values.squeeze(0), logps.topk(beam_size).indices.squeeze(0)
            if next_state is None:
                next_state = model.decoder.step(beam_emb, beam_state, context)
            for value, index in zip(values, indices):
                token = index.item()
                new_score = score + value.item()
                if token == eos_id:
                    finished.append((new_score, tokens + [token], beam_state, beam_emb))
                else:
                    next_emb = model.decoder.embedding(
                        torch.tensor([token], device=src_ids.device))
                    candidates.append((new_score, tokens + [token], next_state, next_emb))
        beams = sorted(candidates, key=lambda b: b[0], reverse=True)[:beam_size]
    all_beams = sorted(beams + finished, key=lambda b: b[0], reverse=True)
    return [tokens for _, tokens, _, _ in all_beams[:beam_size]]