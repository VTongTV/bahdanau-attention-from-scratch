"""bake token ids and vocab into serialized stores."""

from pathlib import Path

import numpy as np
import torch

from nmt.data.filter import filter_by_length, wrap
from nmt.data.split import dev_pairs, test_pairs
from nmt.data.wmt14 import train_pairs
from nmt.vocab.tokenizer import tokenize
from nmt.vocab.vocabulary import Vocab


class Store:
    """in-memory id rows with per-sentence slicing."""

    def __init__(self, src, tgt, src_len, tgt_len):
        self.src = src
        self.tgt = tgt
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.src_off = np.concatenate([[0], np.cumsum(src_len)])
        self.tgt_off = np.concatenate([[0], np.cumsum(tgt_len)])

    def __len__(self) -> int:
        return len(self.src_len)

    def src_row(self, i) -> torch.Tensor:
        """return the source id row for sentence i."""
        a, b = self.src_off[i], self.src_off[i + 1]
        return torch.tensor(self.src[a:b], dtype=torch.long)

    def tgt_row(self, i) -> torch.Tensor:
        """return the target id row for sentence i."""
        a, b = self.tgt_off[i], self.tgt_off[i + 1]
        return torch.tensor(self.tgt[a:b], dtype=torch.long)

    @classmethod
    def load(cls, path) -> "Store":
        """read a serialized store back into memory."""
        d = np.load(path)
        return cls(d["src"], d["tgt"], d["src_len"], d["tgt_len"])


def token_pairs(pairs):
    """yield tokenized pairs."""
    for src, tgt in pairs:
        yield tokenize(src), tokenize(tgt)


def train_filtered(pairs, max_len):
    """yield wrapped token pairs under the max length."""
    for src, tgt in filter_by_length(token_pairs(pairs), max_len):
        yield wrap(src), wrap(tgt)


def eval_wrapped(pairs):
    """yield wrapped token pairs without a length filter."""
    for src, tgt in token_pairs(pairs):
        yield wrap(src), wrap(tgt)


def build_vocabs(pairs, size):
    """count both sides and build the shortlists."""
    src = Vocab(size)
    tgt = Vocab(size)
    for s, t in pairs:
        src.count([s])
        tgt.count([t])
    src.build()
    tgt.build()
    return src, tgt


def save_store(pairs, src_vocab, tgt_vocab, path):
    """write flat id arrays and lengths for a split."""
    src_flat = []
    tgt_flat = []
    src_len = []
    tgt_len = []
    for s, t in pairs:
        src_flat.extend(src_vocab.id(w) for w in s)
        tgt_flat.extend(tgt_vocab.id(w) for w in t)
        src_len.append(len(s))
        tgt_len.append(len(t))
    np.savez(
        path,
        src=np.asarray(src_flat, dtype=np.int32),
        tgt=np.asarray(tgt_flat, dtype=np.int32),
        src_len=np.asarray(src_len, dtype=np.int64),
        tgt_len=np.asarray(tgt_len, dtype=np.int64),
    )


def prepare(config, out_dir, limit=None):
    """build vocabularies and stores for the train dev and test splits."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = list(train_filtered(train_pairs(config.data_dir, limit), config.max_len))
    src_vocab, tgt_vocab = build_vocabs(train, config.vocab_size)
    src_vocab.save(out / "vocab.src")
    tgt_vocab.save(out / "vocab.tgt")
    save_store(train, src_vocab, tgt_vocab, out / "train.npz")
    save_store(eval_wrapped(dev_pairs(config.data_dir)), src_vocab, tgt_vocab, out / "dev.npz")
    save_store(eval_wrapped(test_pairs(config.data_dir)), src_vocab, tgt_vocab, out / "test.npz")