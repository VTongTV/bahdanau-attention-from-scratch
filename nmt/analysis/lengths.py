"""length statistics over corpus subsets."""

from collections import Counter


def length_histogram(lengths, max_len):
    """count sentence lengths from 1 to max_len."""
    counts = Counter(min(length, max_len) for length in lengths)
    return [counts.get(i, 0) for i in range(1, max_len + 1)]


def length_stats(lengths):
    """mean median and spread of a length list."""
    n = len(lengths)
    if n == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0}
    ordered = sorted(lengths)
    mean = sum(ordered) / n
    mid = n // 2
    median = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    var = sum((x - mean) ** 2 for x in ordered) / n
    return {"count": n, "mean": mean, "median": median, "std": var ** 0.5}


def src_lengths(store):
    """source lengths of every row in a store."""
    return [len(store.src_row(i)) for i in range(len(store))]


def tgt_lengths(store):
    """target lengths of every row in a store."""
    return [len(store.tgt_row(i)) for i in range(len(store))]


def ratio_stats(src_lengths, tgt_lengths):
    """stats over the tgt/src length ratio per pair."""
    ratios = [t / s for s, t in zip(src_lengths, tgt_lengths) if s > 0]
    stats = length_stats(ratios)
    stats["min"] = min(ratios) if ratios else 0.0
    stats["max"] = max(ratios) if ratios else 0.0
    return stats


def longest_rows(store, n):
    """indices of the n longest source rows."""
    order = sorted(range(len(store)), key=lambda i: -len(store.src_row(i)))
    return order[:n]