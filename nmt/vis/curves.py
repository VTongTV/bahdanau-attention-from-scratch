"""learning and bleu curve plots from run artifacts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def bleu_curve(bucket_rows, out_path, title="bleu by source length"):
    """line plot of corpus bleu per length bucket."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = [label for label, _, _ in bucket_rows]
    scores = [bleu for _, bleu, _ in bucket_rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(labels)), scores, marker="o")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("source length bucket")
    ax.set_ylabel("corpus bleu")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def learning_curves(csv_path, out_path, title="train and dev nll by epoch"):
    """line plot of train and dev nll over epochs from a run csv."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    epochs, train, dev = [], [], []
    with open(csv_path, encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                epochs.append(int(parts[0]))
                train.append(float(parts[2]))
                dev.append(float(parts[3]))
            except ValueError:
                continue
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train, marker="o", label="train")
    ax.plot(epochs, dev, marker="s", label="dev")
    ax.set_xlabel("epoch")
    ax.set_ylabel("nll")
    ax.set_title(title, fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)