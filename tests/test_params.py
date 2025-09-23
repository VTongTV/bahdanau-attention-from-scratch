"""init tests for orthogonality and distribution spreads."""

import torch

from nmt.config import WEIGHT_INIT_STD, ALIGNMENT_INIT_STD
from nmt.model.params import apply_paper_init, gaussian, orthogonal, zero
from nmt.model.gru import ContextFreeCell
from nmt.model.embedding import SourceEmbedding, TargetEmbedding


def test_orthogonal_is_orthogonal():
    w = torch.empty(24, 24)
    orthogonal(w)
    product = w @ w.t()
    assert torch.allclose(product, torch.eye(24), atol=1e-5)


def test_gaussian_spread():
    w = torch.empty(2000, 500)
    gaussian(w, WEIGHT_INIT_STD)
    assert abs(w.mean().item()) < 0.01
    assert abs(w.std().item() - WEIGHT_INIT_STD) < 0.01 * 0.1


def test_alignment_gaussian_std():
    w = torch.empty(1000, 500)
    gaussian(w, ALIGNMENT_INIT_STD)
    assert abs(w.std().item() - ALIGNMENT_INIT_STD) < ALIGNMENT_INIT_STD * 0.1


def test_zero_initializes_vector():
    v = torch.empty(30)
    zero(v)
    assert torch.all(v == 0)


def test_default_rule_on_vectors():
    m = torch.nn.Linear(10, 20)
    apply_paper_init(m)
    assert torch.all(m.bias == 0)


def test_recurrent_orthogonal_after_model_init():
    cell = ContextFreeCell(input_size=8, hidden_size=16)
    cell.init_parameters()
    for w in cell.recurrent_weights():
        product = w @ w.t()
        assert torch.allclose(product, torch.eye(16), atol=1e-5)


def test_embeddings_init_spread():
    emb = SourceEmbedding(50, 20)
    emb.init_parameters()
    assert abs(emb.table.weight.std().item() - WEIGHT_INIT_STD) < WEIGHT_INIT_STD * 0.2
    tgt = TargetEmbedding(50, 20)
    tgt.init_parameters()
    assert abs(tgt.table.weight.std().item() - WEIGHT_INIT_STD) < WEIGHT_INIT_STD * 0.2


def test_apply_paper_init_tree():
    src = SourceEmbedding(50, 20)
    tgt = TargetEmbedding(50, 20)
    tree = torch.nn.ModuleList([src, tgt])
    apply_paper_init(tree)
    for emb in tree:
        assert abs(emb.table.weight.std().item() - WEIGHT_INIT_STD) < WEIGHT_INIT_STD * 0.2