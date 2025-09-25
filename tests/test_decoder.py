"""decoder tests for state init and context step."""

import torch

from nmt.config import ExperimentConfig
from nmt.model.decoder import Decoder, DecoderCell


def make_decoder():
    config = ExperimentConfig(hidden=20, embedding=10, vocab_size=60)
    return Decoder(config)


def test_initial_state_uses_backward_first():
    dec = make_decoder()
    bwd_first = torch.randn(4, 20)
    s0 = dec.initial_state(bwd_first)
    assert s0.shape == (4, 20)
    assert torch.all((s0 > -1) * (s0 < 1))


def test_w_s_maps_hidden_to_hidden():
    dec = make_decoder()
    assert dec.w_s.weight.shape == (20, 20)


def test_context_cell_step_shape():
    cell = DecoderCell(ExperimentConfig(hidden=20, embedding=10, vocab_size=60))
    emb = torch.randn(3, 10)
    state = torch.zeros(3, 20)
    context = torch.randn(3, 40)
    out = cell(emb, state, context)
    assert out.shape == (3, 20)
    assert torch.all((out > -1) * (out < 1))


def test_context_affects_gates():
    cell = DecoderCell(ExperimentConfig(hidden=20, embedding=10, vocab_size=60))
    emb = torch.randn(3, 10)
    state = torch.randn(3, 20)
    zero_ctx = torch.zeros(3, 40)
    neg_ctx = -50.0 * torch.ones(3, 40)
    with torch.no_grad():
        a = cell(emb, state, zero_ctx)
        b = cell(emb, state, neg_ctx)
    assert not torch.allclose(a, b)


def test_decoder_init_parameters():
    dec = make_decoder()
    dec.init_parameters()
    assert dec.cell.b.abs().max() == 0
    assert torch.allclose(dec.cell.u.weight @ dec.cell.u.weight.T,
                          torch.eye(20), atol=1e-4)