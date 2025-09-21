"""utf-8 normalization for raw text lines."""

import unicodedata

from nmt.config import UNK


def normalize(text: str) -> str:
    """normalize unicode and collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def read_lines(path, normalize_text: bool = True):
    """yield normalized utf-8 lines from a text file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield normalize(line) if normalize_text else line.rstrip("\n")


def fallback_unk(tokens: list, vocab) -> list:
    """map tokens outside the vocab to the unk token."""
    return [t if t in vocab else UNK for t in tokens]