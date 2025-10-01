"""forward-shape tests for the rnnsearch and rnnencdec models."""

import torch

from nmt.config import ExperimentConfig
from nmt.model.rnnencdec import RNNencdec
from nmt.model.rnnsearch import RNNsearch


def make_rnnsearch():
    return RNNsearch(ExperimentConfig(
        hidden=10, embedding=5, vocab_size=30, maxout=6, alignment_hidden=8))


def make_rnnencdec():
    return RNNencdec(ExperimentConfig(
        hidden=10, embedding=5, vocab_size=30, maxout=6, alignment_hidden=8))


def test_rnnsearch_forward_shape():
    model = make_rnnsearch()
    src = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
    tgt = torch.tensor([[1, 8, 9, 0], [1, 8, 9, 10]])
    logits = model(src, tgt)
    assert logits.shape == (2, 3, 30)


def test_rnnsearch_single_sentence():
    model = make_rnnsearch()
    src = torch.tensor([[2, 5, 9]])
    tgt = torch.tensor([[1, 8, 9]])
    logits = model(src, tgt)
    assert logits.shape == (1, 2, 30)


def test_rnnencdec_forward_shape():
    model = make_rnnencdec()
    src = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
    tgt = torch.tensor([[1, 8, 9, 0], [1, 8, 9, 10]])
    logits = model(src, tgt)
    assert logits.shape == (2, 3, 30)


def test_rnnencdec_single_sentence():
    model = make_rnnencdec()
    src = torch.tensor([[2, 5, 9]])
    tgt = torch.tensor([[1, 8, 9]])
    logits = model(src, tgt)
    assert logits.shape == (1, 2, 30)


def test_rnnsearch_init_parameters_runs():
    model = make_rnnsearch()
    model.init_parameters()
    assert model.decoder.cell.b.abs().max() == 0
    assert model.attention.alignment.v_a.abs().max() == 0
    assert model.head.maxout.b.abs().max() == 0


def test_rnnencdec_init_parameters_runs():
    model = make_rnnencdec()
    model.init_parameters()
    assert model.decoder.cell.b.abs().max() == 0
    assert model.head.maxout.b.abs().max() == 0