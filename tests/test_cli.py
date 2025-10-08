"""cli entry point tests."""

from nmt.cli import evaluate, print_bleu


def test_evaluate_identical_corpus(tmp_path, capsys):
    hyps = tmp_path / "hyps.ids"
    refs = tmp_path / "refs.ids"
    hyps.write_text("1 5 6 7 8\n9 10 11\n", encoding="utf-8")
    refs.write_text("1 5 6 7 8\n9 10 11\n", encoding="utf-8")
    score = evaluate(["--hypotheses", str(hyps), "--references", str(refs)])
    assert score == 1.0
    assert "bleu 1.00" in capsys.readouterr().out


def test_evaluate_mismatched_corpus(tmp_path):
    hyps = tmp_path / "hyps.ids"
    refs = tmp_path / "refs.ids"
    hyps.write_text("1 5\n", encoding="utf-8")
    refs.write_text("9 10 11\n", encoding="utf-8")
    assert evaluate(["--hypotheses", str(hyps), "--references", str(refs)]) < 1.0


def test_print_bleu_roundtrip(capsys):
    assert print_bleu(0.25) == 0.25
    assert "bleu 0.25" in capsys.readouterr().out