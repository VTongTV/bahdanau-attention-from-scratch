"""alignment sampling tests."""

import torch

from nmt.analysis.align_samples import greedy_with_align, sample_alignments
from nmt.config import ExperimentConfig
from nmt.model.rnnsearch import RNNsearch


def make_model():
    model = RNNsearch(ExperimentConfig(
        hidden=10, embedding=5, vocab_size=30, maxout=6, alignment_hidden=8))
    model.init_parameters()
    model.eval()
    return model


class FakeStore:
    """minimal store exposing src rows as tensors."""

    def __init__(self, rows):
        self.rows = [torch.tensor(r) for r in rows]

    def src_row(self, i):
        return self.rows[i]

    def __len__(self):
        return len(self.rows)


def test_greedy_with_align_records_weights():
    model = make_model()
    src = torch.tensor([[2, 5, 9, 1]])
    tokens, weights = greedy_with_align(model, src, max_len=10)
    assert weights.shape[0] == len(tokens) - 1
    assert weights.shape[1] == 4
    assert torch.allclose(weights.sum(dim=1), torch.ones(len(tokens) - 1), atol=1e-5)
    assert tokens[0] == 0


def test_greedy_with_align_stops_on_eos():
    model = make_model()
    src = torch.tensor([[2, 5, 9, 1]])
    tokens, _ = greedy_with_align(model, src, max_len=100)
    assert len(tokens) >= 2
    assert tokens[-1] == 1


def test_sample_alignments_returns_samples():
    model = make_model()
    store = FakeStore(rows=[[2, 5, 9, 1], [3, 7, 1]])
    samples = sample_alignments(model, store, [0, 1], max_len=10)
    assert len(samples) == 2
    assert samples[0][0] == 0
    assert samples[0][3].shape[1] == 4