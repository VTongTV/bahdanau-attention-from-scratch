"""frequency statistics for the shortlist cut."""

from nmt.vocab.vocabulary import Vocab


def coverage(vocab: Vocab, tokens) -> float:
    """fraction of tokens that stay in the shortlist."""
    total = 0
    kept = 0
    for line in tokens:
        for t in line:
            total += 1
            if t in vocab:
                kept += 1
    return kept / total if total else 0.0


def unk_rate(vocab: Vocab, tokens) -> float:
    """fraction of tokens mapped to the unk token."""
    total = 0
    unks = 0
    for line in tokens:
        for t in line:
            total += 1
            if t not in vocab:
                unks += 1
    return unks / total if total else 0.0


def summary(vocab: Vocab, tokens) -> dict:
    """return the shortlist stats as a dict."""
    return {
        "vocab_size": len(vocab),
        "coverage": coverage(vocab, tokens),
        "unk_rate": unk_rate(vocab, tokens),
    }