"""paper-format result tables generated from run artifacts."""

from pathlib import Path

from nmt.eval.corpus_bleu import corpus_bleu
from nmt.eval.scorer import parse_ids
from nmt.eval.subsets import no_unk_indices
from nmt.vocab.special import special_ids
from nmt.vocab.vocabulary import Vocab


def load_run_rows(run_dir, data_dir, vocab_path):
    """one row of numbers per completed run directory."""
    run_dir = Path(run_dir)
    out_path = run_dir / "test.npz.out"
    if not out_path.exists():
        return None
    preds = [parse_ids(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    vocab = Vocab.load(vocab_path)
    unk = special_ids(vocab)["unk"]
    srcs, refs = read_store(data_dir, len(preds))
    csv_path = run_dir / "train.csv"
    train_nll, dev_nll, updates, epochs = read_nll(csv_path)
    kept = no_unk_indices(srcs, refs, unk)
    nounk_path = run_dir / "test.npz.nounk.out"
    if nounk_path.exists():
        preds_nounk = [parse_ids(line) for line in
                       nounk_path.read_text(encoding="utf-8").splitlines()]
        no_unk_bleu = corpus_bleu(preds_nounk, [refs[i] for i in kept])
    else:
        no_unk_bleu = corpus_bleu([preds[i] for i in kept], [refs[i] for i in kept])
    return {
        "train_nll": train_nll,
        "dev_nll": dev_nll,
        "updates": updates,
        "epochs": epochs,
        "bleu": corpus_bleu(preds, refs),
        "no_unk_bleu": no_unk_bleu,
        "no_unk_pairs": len(kept),
    }


def read_store(data_dir, count):
    """source and target ids of the first count test rows."""
    from nmt.data.prepare import Store

    store = Store.load(Path(data_dir) / "test.npz")
    n = min(count, len(store))
    return ([store.src_row(i).tolist() for i in range(n)],
            [store.tgt_row(i).tolist() for i in range(n)])


def read_nll(csv_path):
    """final train and dev nll plus totals from a run csv."""
    train = dev = 0.0
    updates = 0
    epochs = 0
    if not Path(csv_path).exists():
        return train, dev, updates, epochs
    with open(csv_path, encoding="utf-8") as fh:
        next(fh, None)
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            epochs = int(parts[0])
            updates = int(parts[1])
            train = float(parts[2])
            dev = float(parts[3])
    return train, dev, updates, epochs


def render_markdown(rows):
    """the paper table 1 and table 2 markdown from run rows."""
    lines = ["| method | all sentences | no unk subset |", "| --- | --- | --- |"]
    for model, rows_by_len in rows.items():
        for max_len, row in rows_by_len.items():
            if row is None:
                lines.append(f"| {model}-{max_len} | | |")
                continue
            lines.append(
                f"| {model}-{max_len} | {row['bleu']:.2f} | {row['no_unk_bleu']:.2f} |"
            )
    return "\n".join(lines)