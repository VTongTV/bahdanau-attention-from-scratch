"""experiment driver: train, decode, score each run in the matrix."""

from pathlib import Path

import torch

from nmt.config import ExperimentConfig
from nmt.data.prepare import Store
from nmt.decode.corpus import decode_all
from nmt.eval.results import load_run_rows
from nmt.train.checkpoint import load_checkpoint
from nmt.train.train import build_model, run
from nmt.utils.device import pick_device


def run_matrix(configs, out_root):
    """train every config in the matrix under its own run directory."""
    root = Path(out_root)
    for config in configs:
        config.run_dir = str(root / config.model / f"max{config.max_len}")
        run(config)


def decode_run(run_dir, data_dir):
    """translate the whole test set with the best checkpoint."""
    run_dir = Path(run_dir)
    checkpoint = run_dir / "checkpoint.best.pt"
    raw = torch.load(checkpoint, weights_only=False)
    config = ExperimentConfig.from_dict(raw["config"])
    model = build_model(config)
    load_checkpoint(checkpoint, model, None, config)
    model.to(pick_device(config.device))
    model.eval()
    store = Store.load(Path(data_dir) / "test.npz")
    decode_all(model, store, config, run_dir / "test.npz.out")


def collect_run(run_dir, data_dir, vocab_path):
    """all report numbers for one completed run."""
    return load_run_rows(run_dir, data_dir, vocab_path)