"""decode harness tests: greedy beam unk and corpus decoding."""

from pathlib import Path

import numpy as np
import torch

from nmt.config import ExperimentConfig
from nmt.data.prepare import Store
from nmt.decode.corpus import decode_all
from nmt.model.rnnsearch import RNNsearch


def make_search():
    config = ExperimentConfig(
        hidden=8, embedding=4, vocab_size=30, maxout=6,
        alignment_hidden=8, max_len=10,
    )
    model = RNNsearch(config)
    model.init_parameters()
    model.eval()
    return model


def make_store(n=5):
    rng = np.random.default_rng(3)
    sl = rng.integers(2, 8, n)
    tl = rng.integers(2, 8, n)
    return Store(
        rng.integers(3, 29, int(sl.sum())),
        rng.integers(3, 29, int(tl.sum())),
        sl.astype(np.int64),
        tl.astype(np.int64),
    )


def test_decode_all_writes_one_line_per_sentence(tmp_path):
    store = make_store(6)
    out = Path(tmp_path) / "out.txt"
    decode_all(make_search(), store, make_search().config, out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    assert all(line.split() for line in lines)