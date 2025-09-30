"""end-to-end baseline training: rnnencdec learns on a toy store."""

import warnings
from pathlib import Path

import numpy as np
import torch

from nmt.config import ExperimentConfig
from nmt.data.prepare import Store
from nmt.model.rnnencdec import RNNencdec
from nmt.train.checkpoint import load_checkpoint, save_checkpoint
from nmt.train.optimizer import Adadelta
from nmt.train.snapshot import BestSnapshot
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


def test_adadelta_statistics_reported():
    lin = torch.nn.Linear(4, 4)
    opt = Adadelta(lin.parameters())
    g, dx, scale = opt.statistics()
    assert g == 0.0 and dx == 0.0 and scale == 0.0
    lin(torch.randn(4, 4)).sum().backward()
    opt.step()
    g, dx, scale = opt.statistics()
    assert g > 0 and dx > 0 and scale > 0


def test_best_snapshot_saves_only_on_improvement(tmp_path):
    calls = []
    snapshot = BestSnapshot(tmp_path / "best.pt", lambda p: calls.append(p))
    assert snapshot.observe(5.0, 10)
    assert not snapshot.observe(6.0, 20)
    assert snapshot.observe(4.0, 30)
    assert len(calls) == 2
    assert calls[0] == tmp_path / "best.pt"
    assert snapshot.best == 4.0
    assert snapshot.best_update == 30


def test_resume_config_fidelity(tmp_path):
    warnings.filterwarnings("ignore")
    torch.manual_seed(5)
    config = ExperimentConfig(
        hidden=10, embedding=5, vocab_size=30, maxout=6,
        alignment_hidden=8, minibatch=4, rebucket_pool=12, seed=5,
    )
    model = RNNencdec(config)
    model.init_parameters()
    optimizer = Adadelta(model.parameters())
    path = Path(tmp_path) / "ckpt.pt"
    save_checkpoint(path, model, optimizer, config, 7)
    other = RNNencdec(ExperimentConfig(hidden=12, embedding=5, vocab_size=30,
                                       maxout=6, alignment_hidden=8))
    cli = ExperimentConfig(hidden=12, embedding=5, vocab_size=30,
                           maxout=6, alignment_hidden=8)
    try:
        load_checkpoint(path, other, None, cli)
        assert False, "different hidden size must raise"
    except ValueError:
        pass