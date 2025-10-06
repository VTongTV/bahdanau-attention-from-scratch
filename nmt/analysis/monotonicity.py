"""attention monotonicity: how much weight stays on the diagonal."""

import torch


def diagonal_mass(weights, band=2):
    """fraction of attention within band steps of the diagonal."""
    t, s = weights.shape
    if t == 0 or s == 0:
        return 0.0
    total = float(weights.sum())
    if total == 0:
        return 0.0
    on_band = 0.0
    for i in range(t):
        lo = max(0, i - band)
        hi = min(s, i + band + 1)
        on_band += float(weights[i, lo:hi].sum())
    return on_band / total


def per_step_argmax(weights):
    """index of the argmax weight for every target step."""
    t, s = weights.shape
    out = []
    for i in range(t):
        if torch.max(weights[i]) > 0:
            out.append(int(torch.argmax(weights[i])))
        else:
            out.append(-1)
    return out


def monotonicity_stats(samples, band=2):
    """aggregate diagonal mass over a list of weight matrices."""
    masses = [diagonal_mass(w, band) for _, _, _, w in samples]
    masses = [m for m in masses if m > 0]
    if not masses:
        return {"count": 0, "mean": 0.0, "median": 0.0}
    ordered = sorted(masses)
    n = len(ordered)
    mid = n // 2
    median = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "count": n,
        "mean": sum(ordered) / n,
        "median": median,
        "min": ordered[0],
        "max": ordered[-1],
    }