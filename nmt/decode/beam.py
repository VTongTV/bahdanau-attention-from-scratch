"""beam search decoding with width and log-prob accumulation."""

import torch

from nmt.decode.translate import prepare_batch
from nmt.decode.unk import mask_unk


def beam_search(model, src_ids, src_mask=None, bos_id=0, eos_id=1, unk_id=2,
                beam_size=10, unk_suppress=False, max_len=100, trace=None):
    """decode one source. prepare the encoder state and run the beam loop."""
    state, context_of = prepare_batch(model, src_ids, src_mask)
    return beam_loop(model, state, context_of, src_mask, bos_id, eos_id, unk_id,
                     beam_size, unk_suppress, max_len, trace)


def beam_loop(model, state, context_of, src_mask=None, bos_id=0, eos_id=1,
              unk_id=2, beam_size=10, unk_suppress=False, max_len=100,
              trace=None):
    """keep the top beam_size paths, all beams in one batched tensor."""
    device = state.device
    emb = model.decoder.embedding(torch.full((1,), bos_id, device=device))
    state = state.contiguous()
    scores = torch.zeros(1, device=device)
    tokens = torch.full((1, 1), bos_id, device=device, dtype=torch.long)
    dead = torch.zeros(1, dtype=torch.bool, device=device)
    finished = []
    for step in range(max_len):
        if dead.all():
            break
        masks = src_mask.expand(len(state), -1) if src_mask is not None else None
        context = context_of(state, masks)
        logits = model.head.forward(state, emb, context)
        if unk_suppress and unk_id is not None:
            logits = mask_unk(logits, unk_id)
        logps = torch.log_softmax(logits, dim=-1)
        logps = logps.masked_fill(dead.unsqueeze(-1), float("-inf"))
        values, indices = logps.topk(beam_size, dim=-1)
        parents = torch.arange(len(state), device=device).unsqueeze(-1) \
            .expand(-1, beam_size).flatten()
        cand_scores = (scores.unsqueeze(-1) + values).flatten()
        cand_tokens = indices.flatten()
        eos = cand_tokens == eos_id
        if eos.any():
            fin_scores = cand_scores[eos].tolist()
            fin_tokens = torch.cat(
                [tokens[parents[eos]], cand_tokens[eos].unsqueeze(-1)],
                dim=-1).tolist()
            finished.extend(zip(fin_scores, fin_tokens))
        alive = ~eos
        non_eos_scores = cand_scores[alive]
        non_eos_tokens = cand_tokens[alive]
        non_eos_parents = parents[alive]
        n = min(len(non_eos_scores), beam_size)
        if n == 0:
            dead = torch.ones(len(state), dtype=torch.bool, device=device)
            continue
        order = torch.argsort(non_eos_scores, descending=True)[:n]
        next_scores = non_eos_scores[order]
        next_tokens = non_eos_tokens[order]
        parent_ids = non_eos_parents[order]
        tokens = torch.cat([tokens[parent_ids], next_tokens.unsqueeze(-1)], dim=-1)
        scores = next_scores
        emb = model.decoder.embedding(next_tokens)
        state = model.decoder.step(emb, state[parent_ids], context[parent_ids])
        dead = torch.zeros(n, dtype=torch.bool, device=device)
        if trace is not None:
            trace(step, [(s, t) for s, t in
                         zip(next_scores.tolist(), tokens.tolist())])
    alive_beams = [(scores[i].item(), tokens[i].tolist())
                   for i in range(len(scores)) if not dead[i]]
    all_beams = sorted(alive_beams + finished, key=lambda b: b[0], reverse=True)
    return [tokens for _, tokens in all_beams[:beam_size]]