"""alignment and attention tests."""

import torch

from nmt.config import ExperimentConfig
from nmt.model.alignment import Alignment
from nmt.model.attention import Attention


def make_attention():
    config = ExperimentConfig(hidden=12, embedding=6, vocab_size=40, alignment_hidden=10)
    return Attention(config)


def make_source():
    torch.manual_seed(7)
    return torch.randn(3, 5, 24)


def test_alignment_score_shape():
    att = make_attention()
    att.cache(make_source())
    state = torch.randn(3, 12)
    scores = att.alignment.score(state)
    assert scores.shape == (3, 5)


def test_cached_equals_direct():
    att = make_attention()
    annotations = make_source()
    att.cache(annotations)
    state = torch.randn(3, 12)
    scores = att.alignment.score(state)
    direct = torch.einsum(
        "btk,k->bt",
        torch.tanh(att.alignment.w_a(state).unsqueeze(1) + att.alignment.u_a(annotations)),
        att.alignment.v_a,
    )
    assert torch.allclose(scores, direct, atol=1e-6)


def test_weights_sum_to_one():
    att = make_attention()
    att.cache(make_source())
    state = torch.randn(3, 12)
    _, weights = att(state)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=1e-6)


def test_padding_gets_zero_weight():
    att = make_attention()
    att.cache(make_source())
    state = torch.randn(3, 12)
    mask = torch.tensor([[True, True, True, False, False]] * 3)
    _, weights = att(state, mask)
    assert torch.all(weights[:, 3:] == 0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=1e-6)


def test_context_dim_matches_annotation():
    att = make_attention()
    att.cache(make_source())
    state = torch.randn(3, 12)
    context, _ = att(state)
    assert context.shape == (3, 24)


def test_context_is_weighted_sum():
    att = make_attention()
    annotations = make_source()
    att.cache(annotations)
    context, weights = att(torch.randn(3, 12))
    manual = torch.einsum("bt,btd->bd", weights, annotations)
    assert torch.allclose(context, manual, atol=1e-6)


def test_alignment_init_spread():
    att = make_attention()
    att.apply_initialization()
    assert att.alignment.v_a.abs().max() == 0.05
    assert att.alignment.w_a.weight.std().item() < 0.01
    assert att.alignment.u_a.weight.std().item() < 0.01