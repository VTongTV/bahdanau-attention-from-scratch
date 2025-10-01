"""evaluation subsets: no-unk corpora and length strata."""


def no_unk_pairs(src_list, pred_list, ref_list, unk):
    """keep triples where neither source nor reference has unk."""
    kept = []
    for src, pred, ref in zip(src_list, pred_list, ref_list):
        if unk in src or unk in ref:
            continue
        kept.append((src, pred, ref))
    return kept