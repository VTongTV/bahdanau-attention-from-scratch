"""evaluation subsets: no-unk corpora and length strata."""

from nmt.eval.corpus_bleu import corpus_bleu


def no_unk_pairs(src_list, pred_list, ref_list, unk):
    """keep triples where neither source nor reference has unk."""
    kept = []
    for src, pred, ref in zip(src_list, pred_list, ref_list):
        if unk in src or unk in ref:
            continue
        kept.append((src, pred, ref))
    return kept


def length_buckets(src_list, pred_list, ref_list, edges=(10, 20, 30, 40)):
    """group pairs by source length and score each bucket."""
    buckets = {}
    for src, pred, ref in zip(src_list, pred_list, ref_list):
        label = bucket_label(len(src), edges)
        buckets.setdefault(label, ([], []))
        buckets[label][0].append(pred)
        buckets[label][1].append(ref)
    rows = []
    for label in sorted(buckets):
        preds, refs = buckets[label]
        rows.append((label, corpus_bleu(preds, refs), len(preds)))
    return rows


def bucket_label(length, edges):
    """the bucket name for a source length."""
    for i, edge in enumerate(edges):
        if length <= edge:
            return f"0-{edge}"
    return f"{edges[-1] + 1}+"