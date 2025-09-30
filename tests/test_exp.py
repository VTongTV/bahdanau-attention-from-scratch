"""experiment driver tests."""

from pathlib import Path

from nmt.config import ExperimentConfig
from nmt.exp.configs import rnnsearch_30
from nmt.exp.runner import run_matrix


def test_rnnsearch_30_config_dims():
    cfg = rnnsearch_30()
    assert cfg.model == "rnnsearch"
    assert cfg.max_len == 30
    assert cfg.hidden == 128
    assert cfg.embedding == 64
    assert cfg.alignment_hidden == 128
    assert cfg.maxout == 64


def test_run_matrix_names_run_dirs(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("nmt.exp.runner.run", lambda c: calls.append(c))
    cfg1 = ExperimentConfig(model="rnnsearch", max_len=30)
    cfg2 = ExperimentConfig(model="rnnencdec", max_len=50)
    run_matrix([cfg1, cfg2], tmp_path)
    assert len(calls) == 2
    assert calls[0].run_dir == str(Path(tmp_path) / "rnnsearch" / "max30")
    assert calls[1].run_dir == str(Path(tmp_path) / "rnnencdec" / "max50")