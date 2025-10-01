"""per-sentence score table export as csv."""

import csv
from pathlib import Path

from nmt.eval.bleu import sentence_bleu


def per_sentence_csv(src_list, pred_list, ref_list, out_path, unk):
    """write one row per triple with sentence bleu and flags."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["src", "pred", "ref", "bleu", "pred_unk", "ref_unk", "no_unk"])
        for src, pred, ref in zip(src_list, pred_list, ref_list):
            writer.writerow([
                " ".join(src),
                " ".join(pred),
                " ".join(ref),
                f"{sentence_bleu(pred, ref):.4f}",
                int(pred.count(unk)),
                int(ref.count(unk)),
                int(unk not in src and unk not in ref),
            ])