"""moses-style tokenizer splitting punctuation from words."""

import re

from nmt.config import UNK

# keep contractions and apostrophes together, split other punctuation
_PUNCT = re.compile(r"([^\w\s'\u2019-]|[\u2019'](?![a-zA-Z]))")
_WHITESPACE = re.compile(r"\s+")


def tokenize(text: str) -> list:
    """split a line into tokens. punctuation becomes separate tokens."""
    text = text.strip()
    text = _PUNCT.sub(r" \1 ", text)
    return _WHITESPACE.split(text.strip())


def tokenize_line(text: str, vocab=None) -> list:
    """tokenize and map unknown tokens to the unk token."""
    tokens = tokenize(text)
    if vocab is None:
        return tokens
    return [t if t in vocab else UNK for t in tokens]