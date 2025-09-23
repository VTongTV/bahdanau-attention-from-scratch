"""gru cell tests for shapes and gate ranges."""

import torch

from nmt.model.gru import ContextFreeCell, GRUCell


def test_context_free_shapes():
    cell = ContextFreeCell(input_size=8, hidden_size=16)
    x = torch.randn(4, 8)
    h = torch.zeros(4, 16)
    h1 = cell(x, h)
    assert h1.shape == (4, 16)
    h2 = cell(x, h1)
    assert h2.shape == (4, 16)


def test_context_cell_shapes():
    cell = GRUCell(input_size=8, hidden_size=16, context_size=24)
    x = torch.randn(4, 8)
    h = torch.zeros(4, 16)
    c = torch.randn(4, 24)
    h1 = cell(x, h, c)
    assert h1.shape == (4, 16)


def test_gates_in_unit_range():
    cell = ContextFreeCell(input_size=8, hidden_size=16)
    cell.init_parameters()
    x = torch.randn(64, 8)
    h = torch.randn(64, 16) * 0.5
    with torch.no_grad():
        z = torch.sigmoid(cell.w_z(x) + cell.u_z(h) + cell.b_z)
        r = torch.sigmoid(cell.w_r(x) + cell.u_r(h) + cell.b_r)
    assert torch.all(z > 0) and torch.all(z < 1)
    assert torch.all(r > 0) and torch.all(r < 1)


def test_hidden_in_tanh_range():
    cell = ContextFreeCell(input_size=8, hidden_size=16)
    cell.init_parameters()
    x = torch.randn(64, 8)
    h = torch.randn(64, 16) * 0.5
    with torch.no_grad():
        z = torch.sigmoid(cell.w_z(x) + cell.u_z(h) + cell.b_z)
        r = torch.sigmoid(cell.w_r(x) + cell.u_r(h) + cell.b_r)
        h_tilde = torch.tanh(cell.w(x) + cell.u(h * r) + cell.b)
    assert torch.all(h_tilde > -1) and torch.all(h_tilde < 1)


def test_state_update_is_blend():
    cell = ContextFreeCell(input_size=8, hidden_size=16)
    x = torch.randn(2, 8)
    h = torch.empty(2, 16).uniform_(-0.5, 0.5)
    out = cell(x, h)
    assert not torch.allclose(out, h)


def test_context_cell_differs_from_free():
    x = torch.randn(2, 8)
    h = torch.zeros(2, 16)
    ctx = GRUCell(input_size=8, hidden_size=16, context_size=16)
    free = ContextFreeCell(input_size=8, hidden_size=16)
    c = torch.randn(2, 16)
    a = ctx(x, h, c)
    b = free(x, h)
    assert not torch.allclose(a, b)


def test_recurrent_weights_list():
    cell = ContextFreeCell(input_size=8, hidden_size=16)
    weights = cell.recurrent_weights()
    assert len(weights) == 3
    for w in weights:
        assert w.shape == (16, 16)