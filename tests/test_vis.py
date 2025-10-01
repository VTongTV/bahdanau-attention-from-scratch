"""alignment heatmap renderer test."""

import numpy as np

from nmt.vis.heatmap import render_heatmap


def test_render_heatmap_saves_png(tmp_path):
    weights = np.array([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]])
    out = tmp_path / "map.png"
    render_heatmap(weights, ["a", "b"], ["x", "y", "z"], out)
    assert out.exists()
    assert out.stat().st_size > 0