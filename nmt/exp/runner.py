"""experiment driver: train, decode, score each run in the matrix."""

from pathlib import Path

import torch

from nmt.config import ExperimentConfig
from nmt.data.iterate import shuffled_order
from nmt.data.prepare import Store
from nmt.decode.corpus import decode_all
from nmt.eval.results import load_run_rows
from nmt.eval.subsets import no_unk_indices
from nmt.train.checkpoint import load_checkpoint
from nmt.train.train import build_model, run
from nmt.train.trainer import Trainer
from nmt.utils.device import pick_device


def run_matrix(configs, out_root):
    """train every config in the matrix under its own run directory."""
    root = Path(out_root)
    for config in configs:
        config.run_dir = str(root / config.model / f"max{config.max_len}")
        run(config)


def decode_run(run_dir, data_dir, out_name="test.npz.out", unk_suppress=False,
               rows=None):
    """translate the test set with the best checkpoint."""
    run_dir = Path(run_dir)
    checkpoint = run_dir / "checkpoint.best.pt"
    raw = torch.load(checkpoint, weights_only=False)
    config = ExperimentConfig.from_dict(raw["config"])
    model = build_model(config)
    load_checkpoint(checkpoint, model, None, config)
    model.to(pick_device(config.device))
    model.eval()
    config.unk_suppress = unk_suppress
    store = Store.load(Path(data_dir) / "test.npz")
    decode_all(model, store, config, run_dir / out_name, rows=rows)


def decode_no_unk(run_dir, data_dir):
    """decode only the no-unk rows with suppression on."""
    run_dir = Path(run_dir)
    all_path = run_dir / "test.npz.out"
    if not all_path.exists():
        return
    store = Store.load(Path(data_dir) / "test.npz")
    srcs = [store.src_row(i).tolist() for i in range(len(store))]
    refs = [store.tgt_row(i).tolist() for i in range(len(store))]
    kept = no_unk_indices(srcs, refs, 2)
    decode_run(run_dir, data_dir, out_name="test.npz.nounk.out",
               unk_suppress=True, rows=kept)


def collect_run(run_dir, data_dir, vocab_path):
    """all report numbers for one completed run."""
    return load_run_rows(run_dir, data_dir, vocab_path)


def dev_nll_run(run_dir, data_dir):
    """dev nll of the best checkpoint, for variance checks."""
    run_dir = Path(run_dir)
    checkpoint = run_dir / "checkpoint.best.pt"
    raw = torch.load(checkpoint, weights_only=False)
    config = ExperimentConfig.from_dict(raw["config"])
    model = build_model(config)
    load_checkpoint(checkpoint, model, None, config)
    model.to(pick_device(config.device))
    trainer = Trainer(model, None, config)
    dev_store = Store.load(Path(data_dir) / "dev.npz")
    order = shuffled_order(len(dev_store), config.seed)
    return trainer.validate(dev_store, order)