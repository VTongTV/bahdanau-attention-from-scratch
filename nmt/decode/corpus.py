"""decode a whole store and write the translations to a file."""

from pathlib import Path

import torch

from nmt.decode.beam import beam_loop
from nmt.decode.translate import prepare_from_annotations
from nmt.decode.unk import drop_unk
from nmt.vocab.special import special_ids

_SLICE = 40


def decode_all(model, store, config, out_path, text_vocab=None,
               drop_unk_flag=False, rows=None):
    """translate every row of a store and write one sentence per line."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    special = special_ids(text_vocab) if text_vocab else {"bos": 0, "eos": 1, "unk": 2}
    device = next(model.parameters()).device
    indices = list(range(len(store))) if rows is None else list(rows)
    with open(out, "w", encoding="utf-8") as fh:
        for start in range(0, len(indices), _SLICE):
            chunk = indices[start: min(start + _SLICE, len(indices))]
            rows_i = [store.src_row(i) for i in chunk]
            lengths = [len(r) for r in rows_i]
            width = max(lengths)
            padded = torch.zeros(len(rows_i), width, dtype=torch.long, device=device)
            for j, r in enumerate(rows_i):
                padded[j, : lengths[j]] = torch.tensor(r, device=device)
            annotations = model.encoder(padded)
            mask = torch.arange(width, device=device).unsqueeze(0) < \
                torch.tensor(lengths, device=device).unsqueeze(1)
            for j in range(len(chunk)):
                state, context_of = prepare_from_annotations(model, annotations[j: j + 1])
                tokens = beam_loop(
                    model, state, context_of, mask[j: j + 1],
                    bos_id=special["bos"], eos_id=special["eos"],
                    unk_id=special["unk"], beam_size=config.beam_size,
                    unk_suppress=config.unk_suppress, max_len=config.max_len,
                )[0]
                if drop_unk_flag:
                    tokens = drop_unk(tokens, special["unk"])
                fh.write(" ".join(str(t) for t in tokens) + "\n")
            fh.flush()