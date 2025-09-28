"""smoke run: a tiny model trains on a toy minibatch end to end."""

import warnings

import numpy as np
import torch

from nmt.config import ExperimentConfig
from nmt.data.prepare import Store
from nmt.model.rnnsearch import RNNsearch
from nmt.train.optimizer import Adadelta
from nmt.train.trainer import Trainer


def toy_config():
    return ExperimentConfig(
        hidden=8, embedding=4, vocab_size=20, maxout=4,
        alignment_hidden=6, minibatch=4, rebucket_pool=16, seed=3,
    )


def toy_store(n=24):
    rng = np.random.default_rng(1)
    sl = rng.integers(2, 6, n)
    tl = rng.integers(2, 6, n)
    return Store(
        rng.integers(1, 19, int(sl.sum())),
        rng.integers(1, 19, int(tl.sum())),
        sl.astype(np.int64),
        tl.astype(np.int64),
    )


def test_smoke_run_one_epoch():
    warnings.filterwarnings("ignore")
    torch.manual_seed(0)
    config = toy_config()
    model = RNNsearch(config)
    model.init_parameters()
    trainer = Trainer(model, Adadelta(model.parameters()), config)
    store = toy_store()
    losses = [trainer.train_step(toy_store_pair(store, i)) for i in range(3)]
    epoch_nll = trainer.run_epoch(store, 0, 99)
    assert all(loss == loss for loss in losses)
    assert epoch_nll == epoch_nll
    assert trainer.updates > 0


def toy_store_pair(store, i):
    from nmt.data.collate import collate

    rows = [i % len(store), (i + 1) % len(store)]
    return collate([store.src_row(r) for r in rows], [store.tgt_row(r) for r in rows])