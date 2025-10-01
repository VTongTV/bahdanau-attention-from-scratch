"""bleu from scratch with n-grams 1-4 and brevity penalty."""

import math
from collections import Counter


def ngrams(seq, n):
    """all n-grams of a token sequence."""
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def clipped_count(pred, ref, n):
    """how many pred n-grams match the reference, capped per type."""
    pred_counts = Counter(ngrams(pred, n))
    ref_counts = Counter(ngrams(ref, n))
    return sum(min(count, ref_counts.get(gram, 0)) for gram, count in pred_counts.items())


def sentence_bleu(pred, ref, max_n=4):
    """bleu for one sentence pair with the brevity penalty."""
    pred_len = len(pred)
    if pred_len == 0:
        return 0.0
    log_precision = 0.0
    counted = 0
    for n in range(1, max_n + 1):
        clipped = clipped_count(pred, ref, n)
        total = pred_len - n + 1
        if total == 0:
            continue
        if clipped == 0:
            return 0.0
        log_precision += math.log(clipped / total)
        counted += 1
    brevity = math.exp(min(0.0, 1 - len(ref) / pred_len))
    return brevity * math.exp(log_precision / max(counted, 1))