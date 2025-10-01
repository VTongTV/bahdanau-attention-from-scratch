"""alignment heatmap renderer saving a figure per sentence."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_heatmap(weights, src_words, tgt_words, out_path, title=""):
    """draw the (tgt x src) weight matrix as a heatmap image."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(len(src_words) * 0.4, 4), max(len(tgt_words) * 0.4, 4)))
    ax.imshow(weights, cmap="Greys", aspect="auto")
    ax.set_xticks(range(len(src_words)))
    ax.set_xticklabels(src_words, rotation=90, fontsize=8)
    ax.set_yticks(range(len(tgt_words)))
    ax.set_yticklabels(tgt_words, fontsize=8)
    ax.set_xlabel("source")
    ax.set_ylabel("target")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)