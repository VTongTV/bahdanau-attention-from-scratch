"""train.py: the cli entry point for a training run."""

import csv
import time
from pathlib import Path

import torch

from nmt.args import config_from_args
from nmt.config import ExperimentConfig
from nmt.data.iterate import shuffled_order
from nmt.data.prepare import Store
from nmt.model.rnnencdec import RNNencdec
from nmt.model.rnnsearch import RNNsearch
from nmt.train.checkpoint import load_checkpoint, save_checkpoint
from nmt.train.earlystop import EarlyStopper
from nmt.train.optimizer import Adadelta
from nmt.train.trainer import Trainer


class CsvLog:
    """append-only csv writer for run metrics."""

    def __init__(self, path):
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["epoch", "update", "train_nll", "val_nll"])

    def row(self, epoch, update, train_nll, val_nll):
        self.writer.writerow([epoch, update, f"{train_nll:.4f}", f"{val_nll:.4f}"])
        self.file.flush()

    def close(self):
        self.file.close()


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
    """the training loop: epochs, dev checks, checkpoints."""
    torch.manual_seed(config.seed)
    train_store, dev_store = load_stores(config)
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(config)
    model.init_parameters()
    optimizer = Adadelta(model.parameters(), config.adadelta_rho, config.adadelta_eps)
    trainer = Trainer(model, optimizer, config)
    start_update = 0
    if config.resume:
        start_update, _ = load_checkpoint(config.resume, model, optimizer, config)
    log = CsvLog(run_dir / "train.csv")
    stopper = EarlyStopper(config.patience)
    dev_order = shuffled_order(len(dev_store), config.seed)
    started = time.time()
    batches_per_epoch = max(len(train_store) // config.minibatch, 1)
    start_epoch = start_update // batches_per_epoch
    for epoch in range(start_epoch, config.epochs):
        train_nll = trainer.run_epoch(train_store, epoch, config.log_every)
        elapsed = time.time() - started
        rate = config.minibatch * trainer.updates / max(elapsed, 1e-9)
        remaining = max(config.epochs - epoch - 1, 0) * batches_per_epoch
        eta = remaining * config.minibatch / max(rate, 1e-9)
        print(f"epoch {epoch} train nll {train_nll:.4f} "
              f"{rate:.0f} sentences/sec eta {eta:.0f}s", flush=True)
        dev_nll = trainer.validate(dev_store, dev_order)
        log.row(epoch, trainer.updates, train_nll, dev_nll)
        save_checkpoint(run_dir / "checkpoint.last.pt", model, optimizer, config, trainer.updates)
        if dev_nll < stopper.best:
            save_checkpoint(run_dir / "checkpoint.best.pt", model, optimizer, config, trainer.updates)
        if stopper.observe(dev_nll):
            print(f"early stop at epoch {epoch} best dev nll {stopper.best:.4f}", flush=True)
            break
    log.close()


if __name__ == "__main__":
    run(config_from_args())