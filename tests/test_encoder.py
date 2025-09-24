"""encoder tests for annotation dims and shared embedding."""

import torch

from nmt.config import ExperimentConfig
from nmt.model.encoder import Encoder


def make_encoder():
    config = ExperimentConfig(hidden=16, embedding=8, vocab_size=50)
    return Encoder(config)


def test_annotation_dims():
    enc = make_encoder()
    ids = torch.randint(0, 50, (3, 7))
    annotations = enc(ids)
    assert annotations.shape == (3, 7, 32)


def test_batch_of_one_sentence():
    enc = make_encoder()
    ids = torch.tensor([[2, 5, 9]])
    annotations = enc(ids)
    assert annotations.shape == (1, 3, 32)


def test_forward_backward_share_embedding():
    enc = make_encoder()
    embedding_params = [p for p in enc.embedding.parameters()]
    assert len(embedding_params) == 1
    assert embedding_params[0].shape == (50, 8)


def test_states_are_separate():
    enc = make_encoder()
    ids = torch.tensor([[1, 2, 3, 4]])
    forward, backward = enc.states(ids)
    assert forward.shape == (1, 4, 16)
    assert backward.shape == (1, 4, 16)
    assert not torch.allclose(forward, backward)