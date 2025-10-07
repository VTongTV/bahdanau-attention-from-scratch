"""reordering study: off-diagonal alignments and displacement stats."""

import torch

from nmt.analysis.monotonicity import diagonal_mass, per_step_argmax


def displacement(weights):
    """mean signed argmax offset from the diagonal per step."""
    offsets = []
    for i, j in enumerate(per_step_argmax(weights)):
        if j >= 0:
            offsets.append(j - i)
    if not offsets:
        return 0.0
    return sum(offsets) / len(offsets)


def reorder_fraction(weights, band=2):
    """share of steps whose argmax lands outside the diagonal band."""
    argmaxes = per_step_argmax(weights)
    if not argmaxes:
        return 0.0
    outside = sum(1 for i, j in enumerate(argmaxes) if j >= 0 and abs(j - i) > band)
    return outside / len(argmaxes)


def shift_matrix(n, k):
    """synthetic weights with all mass shifted k steps off the diagonal."""
    rows = []
    for i in range(n):
        j = i + k
        row = [0.0] * n
        if 0 <= j < n:
            row[j] = 1.0
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def reorder_examples(samples, n=5, band=2):
    """the most reordered samples in a list."""
    ranked = sorted(samples, key=lambda s: -reorder_fraction(s[3], band))
    return ranked[:n]


def reorder_ratio_distribution(weights_list, band=2):
    """histogram of the reorder fraction over many matrices."""
    from collections import Counter

    return Counter(round(reorder_fraction(w, band), 2) for w in weights_list)