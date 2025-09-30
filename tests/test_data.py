"""batch pipeline tests for shapes masks and determinism."""

import numpy as np
import torch

from nmt.config import BOS, EOS, ExperimentConfig
import nmt.data.split as split_mod
from nmt.data.bucket import cut_batches, fetch_pool, rebucket, sort_by_length
from nmt.data.collate import collate
from nmt.data.filter import filter_by_length, unwrap, wrap
from nmt.data.iterate import epoch_batches, shuffled_order
from nmt.data.prepare import Store, eval_wrapped, prepare, save_store, train_filtered
from nmt.data.split import dev_pairs
from nmt.vocab.vocabulary import Vocab


def make_store(n=1000):
    rng = np.random.default_rng(7)
    src_flat = []
    tgt_flat = []
    src_len = []
    tgt_len = []
    for _ in range(n):
        sl = int(rng.integers(3, 12))
        tl = int(rng.integers(3, 12))
        src_flat.extend(rng.integers(0, 50, size=sl).tolist())
        tgt_flat.extend(rng.integers(0, 50, size=tl).tolist())
        src_len.append(sl)
        tgt_len.append(tl)
    return Store(
        np.asarray(src_flat, dtype=np.int32),
        np.asarray(tgt_flat, dtype=np.int32),
        np.asarray(src_len, dtype=np.int64),
        np.asarray(tgt_len, dtype=np.int64),
    )


def test_split_counts():
    dev = list(dev_pairs("data/wmt14"))
    test = list(split_mod.test_pairs("data/wmt14"))
    assert len(dev) == 6003
    assert len(test) == 3003


def test_filter_and_wrap():
    pairs = [(["a"], ["c", "d"]), (["a", "b"], ["d"]), (["a", "b", "c"], ["d"])]
    kept = list(filter_by_length(pairs, 2))
    assert kept == [(["a"], ["c", "d"]), (["a", "b"], ["d"])]
    assert wrap(["a", "b"]) == [BOS, "a", "b", EOS]
    assert unwrap(wrap(["a", "b"])) == ["a", "b"]


def test_rebucket_paper_recipe():
    n = 1600
    lengths = np.arange(n) % 30 + 1
    indices = list(range(n))
    pool, start = fetch_pool(indices, 0, 1600)
    assert start == 1600
    batches = rebucket(pool, lengths, batch_size=80)
    assert len(batches) == 20
    for b in batches:
        assert len(b) == 80
    ordered = sort_by_length(pool, lengths)
    for a, b in zip(ordered, ordered[1:]):
        assert lengths[a] <= lengths[b]


def test_rebucket_deterministic():
    rng = np.random.default_rng(3)
    lengths = rng.integers(1, 50, size=500)
    indices = list(range(500))
    a = rebucket(indices, lengths, batch_size=80)
    b = rebucket(indices, lengths, batch_size=80)
    assert a == b


def test_collate_shapes_and_masks():
    src_rows = [torch.tensor([0, 3, 4, 1]), torch.tensor([0, 5, 1]), torch.tensor([0, 1])]
    tgt_rows = [torch.tensor([0, 6, 1]), torch.tensor([0, 7, 8, 1])]
    src, src_mask, tgt, tgt_mask = collate(src_rows, tgt_rows)
    assert src.shape == (3, 4)
    assert tgt.shape == (2, 4)
    assert src_mask.shape == src.shape
    assert tgt_mask.shape == tgt.shape
    assert src_mask[0].sum() == 4
    assert src_mask[1].sum() == 3
    assert src_mask[2].sum() == 2
    assert tgt_mask[0].sum() == 3
    assert tgt_mask[1].sum() == 4
    assert bool(src_mask[0, 3]) and not bool(src_mask[1, 3])


def test_iterate_deterministic_and_full_coverage():
    store = make_store(300)
    order = shuffled_order(len(store), seed=42)
    batches = list(epoch_batches(store, order, batch_size=80, pool_size=1600))
    seen = []
    for src, src_mask, tgt, tgt_mask in batches:
        assert src.shape == src_mask.shape
        assert tgt.shape == tgt_mask.shape
        assert src.shape[0] <= 80
        seen.append(src.shape[0])
    assert sum(seen) == 300
    assert max(seen) == 80
    order_b = shuffled_order(len(store), seed=42)
    batches_b = list(epoch_batches(store, order_b, batch_size=80, pool_size=1600))
    a = [b[0].tolist() for b in batches]
    c = [b[0].tolist() for b in batches_b]
    assert a == c


def test_store_roundtrip(tmp_path):
    store = make_store(50)
    path = tmp_path / "train.npz"
    pairs = [(store.src_row(i).tolist(), store.tgt_row(i).tolist()) for i in range(50)]
    src_vocab = Vocab(30)
    tgt_vocab = Vocab(30)
    src_vocab.count([p[0] for p in pairs])
    tgt_vocab.count([p[1] for p in pairs])
    src_vocab.build()
    tgt_vocab.build()
    save_store(pairs, src_vocab, tgt_vocab, path)
    loaded = Store.load(path)
    assert len(loaded) == 50
    for i, (s, t) in enumerate(pairs):
        assert loaded.src_row(i).tolist() == [src_vocab.id(w) for w in s]
        assert loaded.tgt_row(i).tolist() == [tgt_vocab.id(w) for w in t]


def test_train_filtered_and_eval_wrapped():
    src_vocab = Vocab(10)
    tgt_vocab = Vocab(10)
    train = list(train_filtered([("a b c", "x y"), ("a b", "x y")], 2))
    assert train == [(wrap(["a", "b"]), wrap(["x", "y"]))]
    ev = list(eval_wrapped([("a b c", "x y")]))
    assert ev == [(wrap(["a", "b", "c"]), wrap(["x", "y"]))]
    src_vocab.count([[BOS, "a", "b", EOS]])
    tgt_vocab.count([[BOS, "x", "y", EOS]])
    src_vocab.build()
    tgt_vocab.build()


def test_config_reads_defaults():
    c = ExperimentConfig()
    assert c.minibatch == 80
    assert c.rebucket_pool == 1600


def test_iterate_carries_tail_across_pools():
    store = make_store(250)
    order = shuffled_order(len(store), seed=1)
    batches = list(epoch_batches(store, order, batch_size=60, pool_size=200))
    seen = sum(b[0].shape[0] for b in batches)
    assert seen == 250
    assert all(b[0].shape[0] <= 60 for b in batches)


def test_prepare_test_mode_slices(tmp_path):
    config = ExperimentConfig(test_mode=True, vocab_size=100, max_len=12)
    prepare(config, tmp_path)
    train = Store.load(tmp_path / "train.npz")
    dev = Store.load(tmp_path / "dev.npz")
    test = Store.load(tmp_path / "test.npz")
    assert len(train) <= 640
    assert len(dev) == 128
    assert len(test) == 128