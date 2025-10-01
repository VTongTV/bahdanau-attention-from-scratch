"""scorer aggregating bleu length and unk stats over decoded output."""

from nmt.eval.corpus_bleu import corpus_bleu
from nmt.eval.subsets import no_unk_pairs


def parse_ids(line):
    """one int per token on a line."""
    return [int(x) for x in line.split()]


def score_output(src_list, pred_list, ref_list, unk):
    """aggregate bleu and length and unk stats over triples."""
    bleu = corpus_bleu(pred_list, ref_list)
    kept = no_unk_pairs(src_list, pred_list, ref_list, unk)
    no_unk_bleu = corpus_bleu([p for _, p, _ in kept], [r for _, _, r in kept])
    return {
        "bleu": bleu,
        "no_unk_bleu": no_unk_bleu,
        "no_unk_pairs": len(kept),
        "pred_len": sum(map(len, pred_list)) / max(len(pred_list), 1),
        "ref_len": sum(map(len, ref_list)) / max(len(ref_list), 1),
        "unk_preds": sum(p.count(unk) for p in pred_list),
        "unk_refs": sum(r.count(unk) for r in ref_list),
    }