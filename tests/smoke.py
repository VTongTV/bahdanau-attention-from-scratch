"""smoke run: a tiny model trains on a toy minibatch end to end."""

import warnings

import numpy as np
import torch

from nmt.config import ExperimentConfig
from nmt.data.prepare import Store
from nmt.decode.beam import beam_search
from nmt.decode.unk import drop_unk, mask_unk
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


def trained_model():
    """a tiny rnnsearch trained on the toy store for decode contracts."""
    warnings.filterwarnings("ignore")
    torch.manual_seed(0)
    config = toy_config()
    model = RNNsearch(config)
    model.init_parameters()
    trainer = Trainer(model, Adadelta(model.parameters()), config)
    store = toy_store()
    for i in range(6):
        trainer.train_step(toy_store_pair(store, i))
    model.eval()
    return model, store


def test_smoke_beam_returns_width_hypotheses():
    model, store = trained_model()
    src = store.src_row(0).reshape(1, -1)
    beams = beam_search(model, src, beam_size=4, max_len=20)
    assert 1 <= len(beams) <= 4
    assert all(len(tokens) >= 1 for tokens in beams)


def test_smoke_beam_stops_all_on_eos():
    model, store = trained_model()
    src = store.src_row(1).reshape(1, -1)
    beams = beam_search(model, src, beam_size=3, max_len=8)
    assert all(len(tokens) <= 9 for tokens in beams)
    assert all(tokens.count(1) <= 1 for tokens in beams)
    assert all(tokens[-1] != 1 or tokens.count(1) == 1 for tokens in beams)


def test_smoke_beam_trace_callback_runs():
    model, store = trained_model()
    src = store.src_row(2).reshape(1, -1)
    steps = []
    beam_search(model, src, beam_size=3, max_len=15,
                trace=lambda step, beams: steps.append(step))
    assert len(steps) >= 1


def test_smoke_unk_guard_never_emits_unk():
    model, store = trained_model()
    src = store.src_row(3).reshape(1, -1)
    beams = beam_search(model, src, beam_size=4, unk_suppress=True,
                        unk_id=2, max_len=30)
    assert all(2 not in tokens for tokens in beams)


def test_smoke_drop_unk_removes_unk_tokens():
    assert drop_unk([1, 2, 3, 2, 4], 2) == [1, 3, 4]


def test_smoke_mask_unk_sets_inf():
    logits = torch.zeros(1, 5)
    masked = mask_unk(logits, 2)
    assert masked[0, 2].item() == float("-inf")
    assert masked[0, 0].item() == 0.0