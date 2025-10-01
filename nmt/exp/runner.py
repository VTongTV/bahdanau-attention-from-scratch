"""experiment driver looping model types and max lengths."""

from pathlib import Path

from nmt.train.train import run


def run_matrix(configs, out_root):
    """run every config in the matrix under its own run directory."""
    root = Path(out_root)
    for config in configs:
        config.run_dir = str(root / config.model / f"max{config.max_len}")
        run(config)