"""wmt14 en-fr subset loader with a configurable corpus size."""

from nmt.data.parallel import load_pairs, pair_files


def train_pairs(data_dir, limit=None):
    """yield raw pairs from the training corpus slice."""
    src, tgt = pair_files(data_dir, "train")
    return load_pairs(src, tgt, limit=limit)


def corpus_size(data_dir):
    """count the pairs in the training corpus."""
    src, _ = pair_files(data_dir, "train")
    with open(src, encoding="utf-8") as f:
        return sum(1 for _ in f)