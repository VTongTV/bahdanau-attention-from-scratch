"""experiment driver tests."""

from pathlib import Path

import numpy as np

from nmt.config import ExperimentConfig
from nmt.eval.results import load_run_rows, render_markdown
from nmt.exp.configs import load_matrix, rnnencdec_30, rnnsearch_30
from nmt.exp.runner import run_matrix


def test_rnnsearch_30_config_dims():
    cfg = rnnsearch_30()
    assert cfg.model == "rnnsearch"
    assert cfg.max_len == 30
    assert cfg.hidden == 128
    assert cfg.embedding == 64
    assert cfg.alignment_hidden == 128
    assert cfg.maxout == 64


def test_rnnencdec_30_matches_search_dims():
    base = rnnsearch_30()
    cfg = rnnencdec_30()
    assert cfg.model == "rnnencdec"
    assert cfg.max_len == base.max_len
    assert cfg.hidden == base.hidden
    assert cfg.embedding == base.embedding
    assert cfg.vocab_size == base.vocab_size


def test_run_matrix_names_run_dirs(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("nmt.exp.runner.run", lambda c: calls.append(c))
    cfg1 = ExperimentConfig(model="rnnsearch", max_len=30)
    cfg2 = ExperimentConfig(model="rnnencdec", max_len=50)
    run_matrix([cfg1, cfg2], tmp_path)
    assert len(calls) == 2
    assert calls[0].run_dir == str(Path(tmp_path) / "rnnsearch" / "max30")
    assert calls[1].run_dir == str(Path(tmp_path) / "rnnencdec" / "max50")


def test_collect_run_reads_artifacts(tmp_path):
    np.savez(tmp_path / "test.npz",
             src=np.array([1, 3, 4, 5], dtype=np.int64),
             tgt=np.array([1, 5, 6, 7, 8], dtype=np.int64),
             src_len=np.array([4], dtype=np.int64),
             tgt_len=np.array([5], dtype=np.int64))
    (tmp_path / "test.npz.out").write_text("1 5 6 7 8\n", encoding="utf-8")
    (tmp_path / "train.csv").write_text(
        "epoch,update,train_nll,val_nll\n9,3920,3.85,3.95\n", encoding="utf-8")
    (tmp_path / "vocab.tgt").write_text("<s>\n</s>\n<unk>\n", encoding="utf-8")
    row = load_run_rows(tmp_path, tmp_path, tmp_path / "vocab.tgt")
    assert row is not None
    assert row["epochs"] == 9
    assert row["bleu"] == 1.0
    assert row["no_unk_bleu"] == 1.0
    assert row["updates"] == 3920


def test_render_markdown_rows_and_empty():
    rows = {"rnnencdec": {"30": {"bleu": 13.93, "no_unk_bleu": 24.19},
                          "50": None}}
    text = render_markdown(rows)
    assert "| rnnencdec-30 | 13.93 | 24.19 |" in text
    assert "| rnnencdec-50 | | |" in text


def test_load_matrix_reads_yaml(tmp_path):
    yaml_path = tmp_path / "experiments.yaml"
    yaml_path.write_text(
        "runs:\n"
        "  - model: rnnsearch\n"
        "    max_len: 30\n"
        "    hidden: 128\n"
        "  - model: rnnencdec\n"
        "    max_len: 50\n"
        "    hidden: 128\n",
        encoding="utf-8")
    configs = load_matrix(yaml_path)
    assert len(configs) == 2
    assert configs[0].model == "rnnsearch"
    assert configs[1].max_len == 50