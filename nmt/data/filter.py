"""sentence filtering by max length and bos/eos wrapping."""

from nmt.config import BOS, EOS


def token_len(tokens) -> int:
    """number of tokens in a token list."""
    return len(tokens)


def filter_by_length(pairs, max_len):
    """yield pairs where both sides fit the max length."""
    for src, tgt in pairs:
        if len(src) <= max_len and len(tgt) <= max_len:
            yield src, tgt


def wrap(tokens) -> list:
    """wrap a token list with bos and eos."""
    return [BOS] + tokens + [EOS]


def unwrap(tokens) -> list:
    """strip bos and eos from a token list."""
    out = [t for t in tokens if t not in (BOS, EOS)]
    return out