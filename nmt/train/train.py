"""train.py: the cli entry point for a training run."""

from pathlib import Path

import torch

from nmt.args import config_from_args
from nmt.config import ExperimentConfig
from nmt.data.prepare import Store
from nmt.model.rnnencdec import RNNencdec
from nmt.model.rnnsearch import RNNsearch
from nmt.train.checkpoint import save_checkpoint
from nmt.train.optimizer import Adadelta
from nmt.train.trainer import Trainer


def build_model(config: ExperimentConfig):
    """instantiate the model named by the config."""
    if config.model == "rnnsearch":
        return RNNsearch(config)
    return RNNencdec(config)


def load_stores(config: ExperimentConfig):
    """read the train and dev stores from the data directory."""
    base = Path(config.data_dir)
    return Store.load(base / "train.npz"), Store.load(base / "dev.npz")


def run(config: ExperimentConfig) -> None:
    """the training loop: epochs, checks, checkpoints."""
    torch.manual_seed(config.seed)
    train_store, _ = load_stores(config)
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(config)
    model.init_parameters()
    optimizer = Adadelta(model.parameters(), config.adadelta_rho, config.adadelta_eps)
    trainer = Trainer(model, optimizer, config)
    for epoch in range(config.epochs):
        train_nll = trainer.run_epoch(train_store, epoch, config.log_every)
        print(f"epoch {epoch} train nll {train_nll:.4f}", flush=True)
        save_checkpoint(run_dir / "checkpoint.last.pt", model, optimizer, config, trainer.updates)


if __name__ == "__main__":
    run(config_from_args())