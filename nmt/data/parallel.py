"""raw parallel line-pair loader."""

from pathlib import Path


def load_pairs(src_path, tgt_path, limit=None):
    """yield (source, target) string pairs from two aligned files."""
    with open(src_path, encoding="utf-8") as fs, open(tgt_path, encoding="utf-8") as ft:
        for i, (s, t) in enumerate(zip(fs, ft)):
            if limit is not None and i >= limit:
                break
            yield s.rstrip("\n"), t.rstrip("\n")


def pair_files(data_dir, split="train"):
    """return the source and target paths for a split."""
    d = Path(data_dir)
    if split == "train":
        return d / "training" / "news-commentary-v9.fr-en.en", d / "training" / "news-commentary-v9.fr-en.fr"
    if split == "dev":
        return d / "dev" / "newstest2012.en", d / "dev" / "newstest2012.fr"
    if split == "test":
        return d / "test" / "newstest2014-fren-src.en.sgm", d / "test" / "newstest2014-fren-src.fr.sgm"
    raise ValueError(f"unknown split {split}")