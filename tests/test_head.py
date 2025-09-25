"""maxout head tests for pair pooling and vocabulary logits."""

import torch

from nmt.config import ExperimentConfig
from nmt.model.head import DeepHead, MaxoutHead


def make_head():
    config = ExperimentConfig(hidden=12, embedding=6, vocab_size=40, maxout=7)
    return DeepHead(config)


def test_maxout_pools_adjacent_pairs():
    config = ExperimentConfig(hidden=12, embedding=6, vocab_size=40, maxout=7)
    head = MaxoutHead(config)
    state = torch.randn(2, 12)
    embedding = torch.randn(2, 6)
    context = torch.randn(2, 24)
    pooled = head(state, embedding, context)
    assert pooled.shape == (2, 7)
    pre = head.u_o(state) + head.v_o(embedding) + head.c_o(context) + head.b
    manual = torch.max(pre.unflatten(-1, (-1, 2)), dim=-1).values
    assert torch.allclose(pooled, manual, atol=1e-6)


def test_logits_shape():
    head = make_head()
    state = torch.randn(3, 12)
    embedding = torch.randn(3, 6)
    context = torch.randn(3, 24)
    assert head(state, embedding, context).shape == (3, 40)


def test_log_probs_normalize():
    head = make_head()
    state = torch.randn(3, 12)
    embedding = torch.randn(3, 6)
    context = torch.randn(3, 24)
    lp = head.log_probs(state, embedding, context)
    assert torch.allclose(lp.exp().sum(dim=-1), torch.ones(3), atol=1e-6)


def test_head_init_parameters():
    head = make_head()
    head.init_parameters()
    assert head.maxout.b.abs().max() == 0
    assert head.w_o.weight.std().item() < 0.1
    assert head.w_o.bias is None