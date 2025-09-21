"""stable ids for the special tokens."""

from nmt.config import BOS, EOS, UNK


def special_ids(vocab) -> dict:
    """return the fixed ids for bos eos and unk."""
    return {
        "bos": vocab.stoi[BOS],
        "eos": vocab.stoi[EOS],
        "unk": vocab.stoi[UNK],
    }


def is_special(token: str) -> bool:
    """true when the token is one of the special tokens."""
    return token in (BOS, EOS, UNK)