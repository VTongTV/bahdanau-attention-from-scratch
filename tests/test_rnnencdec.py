"""rnnencdec baseline tests for a fixed context."""

import torch

from nmt.config import ExperimentConfig
from nmt.model.rnnencdec import RNNencdec


def make_model():
    config = ExperimentConfig(hidden=10, embedding=5, vocab_size=30, maxout=6)
    return RNNencdec(config)


def test_forward_shape():
    model = make_model()
    src = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
    tgt = torch.tensor([[1, 8, 9, 0], [1, 8, 9, 10]])
    logits = model(src, tgt)
    assert logits.shape == (2, 4, 30)


def test_no_attention_module():
    model = make_model()
    assert not hasattr(model, "attention")
    assert not hasattr(model, "alignment")


def test_decoder_starts_from_last_forward():
    model = make_model()
    src = torch.tensor([[1, 2, 3, 4, 5]])
    tgt = torch.tensor([[1, 8, 9]])
    model.encoder.train(False)
    forward_last = model.encoder(src)[:, -1, :10]
    spy = {"state": None}

    def watch(backward_first):
        spy["state"] = backward_first
        return model.decoder.cell(torch.zeros(1, 5), torch.zeros(1, 10), torch.zeros(1, 20))

    model.decoder.initial_state = watch
    model(src, tgt)
    assert torch.allclose(spy["state"], forward_last, atol=1e-6)


def test_init_parameters_runs():
    model = make_model()
    model.init_parameters()
    assert model.decoder.cell.b.abs().max() == 0