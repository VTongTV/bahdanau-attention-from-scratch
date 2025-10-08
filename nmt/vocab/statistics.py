"""frequency stats over a corpus and shortlist coverage."""

from collections import Counter


def token_freqs(tokens_by_line):
    """count token occurrences over a list of token lists."""
    counter = Counter()
    for line in tokens_by_line:
        counter.update(line)
    return counter


def type_token_ratio(counter):
    """distinct types over running tokens."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return len(counter) / total


def coverage(counter, vocab_tokens):
    """share of running tokens that the shortlist covers."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    known = sum(count for token, count in counter.items() if token in vocab_tokens)
    return known / total


def oov_rate(counter, vocab_tokens):
    """share of running tokens outside the shortlist."""
    return 1.0 - coverage(counter, vocab_tokens)