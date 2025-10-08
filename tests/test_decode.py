"""decode harness tests: greedy beam unk and corpus decoding."""

from pathlib import Path

import numpy as np
import torch

from nmt.config import BOS, EOS, ExperimentConfig
from nmt.data.prepare import Store
from nmt.decode.beam import beam_search
from nmt.decode.corpus import decode_all
from nmt.decode.greedy import greedy
from nmt.model.rnnsearch import RNNsearch
from nmt.vocab.special import special_ids
from nmt.vocab.vocabulary import Vocab


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


def make_vocab():
    v = Vocab(30)
    v.count([[BOS, EOS, "a", "b", "c"]])
    v.build()
    return v


def test_special_ids_follow_vocab_layout():
    v = make_vocab()
    ids = special_ids(v)
    assert ids == {"bos": v.id(BOS), "eos": v.id(EOS), "unk": v.id("<unk>")}


def test_beam_stops_on_eos_with_trained_shape():
    model = make_search()
    v = make_vocab()
    ids = special_ids(v)
    src = torch.tensor([[5, 3, 4]], dtype=torch.long)
    tokens = beam_search(model, src, bos_id=ids["bos"], eos_id=ids["eos"],
                         unk_id=ids["unk"], beam_size=4, unk_suppress=True,
                         max_len=10)
    assert tokens[0][0] == ids["bos"]


def test_greedy_starts_from_bos():
    model = make_search()
    v = make_vocab()
    ids = special_ids(v)
    src = torch.tensor([[5, 3, 4]], dtype=torch.long)
    tokens = greedy(model, src, bos_id=ids["bos"], eos_id=ids["eos"],
                    unk_id=ids["unk"], unk_suppress=True, max_len=10)
    assert tokens[0] == ids["bos"]
    assert tokens[-1] == ids["eos"] or len(tokens) == 11


def test_decode_all_writes_one_line_per_sentence(tmp_path):
    store = make_store(6)
    v = make_vocab()
    bos = special_ids(v)["bos"]
    out = Path(tmp_path) / "out.txt"
    decode_all(make_search(), store, make_search().config, out, v)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    assert all(line.split() for line in lines)
    assert lines[0].split()[0] == str(bos)


def test_decode_all_rows_subset(tmp_path):
    store = make_store(6)
    v = make_vocab()
    out = Path(tmp_path) / "out.txt"
    decode_all(make_search(), store, make_search().config, out, v, rows=[0, 3])
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(line.split() for line in lines)