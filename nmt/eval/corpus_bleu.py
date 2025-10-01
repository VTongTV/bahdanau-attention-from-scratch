"""corpus-level bleu matching the paper reporting."""

import math

from nmt.eval.bleu import clipped_count, ngrams


def corpus_bleu(preds, refs, max_n=4):
    """bleu over a whole corpus with clipped n-gram totals."""
    total_candidate = [0] * max_n
    total_clipped = [0] * max_n
    pred_len = 0
    ref_len = 0
    for pred, ref in zip(preds, refs):
        pred_len += len(pred)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            total_candidate[n - 1] += max(len(pred) - n + 1, 0)
            total_clipped[n - 1] += clipped_count(pred, ref, n)
    if pred_len == 0:
        return 0.0
    log_precision = 0.0
    counted = 0
    for n in range(max_n):
        if total_candidate[n] == 0:
            continue
        if total_clipped[n] == 0:
            return 0.0
        log_precision += math.log(total_clipped[n] / total_candidate[n])
        counted += 1
    brevity = math.exp(min(0.0, 1 - ref_len / pred_len))
    return brevity * math.exp(log_precision / max(counted, 1))