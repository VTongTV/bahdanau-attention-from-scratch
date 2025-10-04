"""paper-format result tables generated from run artifacts."""

from pathlib import Path

from nmt.eval.scorer import parse_ids, score_output
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
    scored = score_output(srcs, preds, refs, unk)
    return {
        "train_nll": train_nll,
        "dev_nll": dev_nll,
        "updates": updates,
        "epochs": epochs,
        "bleu": scored["bleu"],
        "no_unk_bleu": scored["no_unk_bleu"],
        "no_unk_pairs": scored["no_unk_pairs"],
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