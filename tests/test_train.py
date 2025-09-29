"""end-to-end baseline training: rnnencdec learns on a toy store."""

import warnings

import numpy as np
import torch

from nmt.config import ExperimentConfig
from nmt.data.prepare import Store
from nmt.model.rnnencdec import RNNencdec
from nmt.train.optimizer import Adadelta
from nmt.train.trainer import Trainer


def test_rnnencdec_trains_end_to_end():
    warnings.filterwarnings("ignore")
    torch.manual_seed(5)
    config = ExperimentConfig(
        hidden=10, embedding=5, vocab_size=30, maxout=6,
        alignment_hidden=8, minibatch=4, rebucket_pool=12, seed=5,
    )
    rng = np.random.default_rng(2)
    n = 12
    sl = rng.integers(2, 7, n)
    tl = rng.integers(2, 7, n)
    store = Store(
        rng.integers(1, 29, int(sl.sum())),
        rng.integers(1, 29, int(tl.sum())),
        sl.astype(np.int64),
        tl.astype(np.int64),
    )
    model = RNNencdec(config)
    model.init_parameters()
    trainer = Trainer(model, Adadelta(model.parameters()), config)
    before = trainer.run_epoch(store, 0, 99)
    after = trainer.run_epoch(store, 1, 99)
    assert after < before