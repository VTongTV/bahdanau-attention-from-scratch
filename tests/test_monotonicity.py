"""monotonicity and reorder tests."""

import torch

from nmt.analysis.monotonicity import (
    diagonal_mass,
    monotonicity_stats,
    per_step_argmax,
)
from nmt.analysis.reorder import (
    displacement,
    reorder_examples,
    reorder_fraction,
    reorder_ratio_distribution,
    shift_matrix,
)


def test_diagonal_mass_identity():
    w = torch.eye(4, dtype=torch.float32)
    assert abs(diagonal_mass(w, band=0) - 1.0) < 1e-6


def test_diagonal_mass_antidiagonal():
    w = torch.zeros(3, 3)
    w[0, 2] = 1.0
    w[1, 1] = 1.0
    w[2, 0] = 1.0
    assert abs(diagonal_mass(w, band=0) - 1.0 / 3.0) < 1e-6


def test_diagonal_mass_empty():
    assert diagonal_mass(torch.zeros(0, 3), band=0) == 0.0


def test_per_step_argmax_known():
    w = torch.tensor([[0.2, 0.8], [0.9, 0.1], [0.0, 0.0]])
    assert per_step_argmax(w) == [1, 0, -1]


def test_monotonicity_stats_known():
    samples = [(0, [], [], torch.eye(3, dtype=torch.float32)),
               (1, [], [], torch.eye(3, dtype=torch.float32))]
    stats = monotonicity_stats(samples, band=0)
    assert stats["count"] == 2
    assert abs(stats["mean"] - 1.0) < 1e-6


def test_monotonicity_stats_empty():
    stats = monotonicity_stats([], band=0)
    assert stats["count"] == 0


def test_displacement_shifted():
    w = shift_matrix(4, 1)
    assert abs(displacement(w) - 1.0) < 1e-6


def test_reorder_fraction_shifted():
    w = shift_matrix(4, 3)
    outside = reorder_fraction(w, band=1)
    assert abs(outside - 0.25) < 1e-9


def test_shift_matrix_diagonal():
    assert shift_matrix(3, 0).sum().item() == 3.0


def test_shift_matrix_out_of_range():
    assert shift_matrix(3, 5).sum().item() == 0.0


def test_reorder_examples_ranks_by_fraction():
    samples = [(0, [], [], shift_matrix(4, 0)),
               (1, [], [], shift_matrix(4, 2))]
    top = reorder_examples(samples, n=1, band=1)
    assert top[0][0] == 1


def test_reorder_ratio_distribution_keys():
    dist = reorder_ratio_distribution([shift_matrix(3, 0), shift_matrix(3, 1)])
    assert sum(dist.values()) == 2