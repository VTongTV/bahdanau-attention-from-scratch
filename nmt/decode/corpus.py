"""decode a whole store and write the translations to a file."""

from pathlib import Path

from nmt.decode.beam import beam_search
from nmt.vocab.special import special_ids


def decode_all(model, store, config, out_path, text_vocab=None):
    """translate every row of a store and write one sentence per line."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    special = special_ids(text_vocab) if text_vocab else {"bos": 0, "eos": 1, "unk": 2}
    with open(out, "w", encoding="utf-8") as fh:
        for i in range(len(store)):
            src = store.src_row(i).reshape(1, -1)
            tokens = beam_search(
                model,
                src,
                bos_id=special["bos"],
                eos_id=special["eos"],
                unk_id=special["unk"],
                beam_size=config.beam_size,
                unk_suppress=config.unk_suppress,
                max_len=config.max_len,
            )[0]
            fh.write(" ".join(str(t) for t in tokens) + "\n")
        fh.flush()