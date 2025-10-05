"""curve plot tests."""

from nmt.vis.curves import bleu_curve, learning_curves


def test_bleu_curve_saves_png(tmp_path):
    rows = [("0-10", 15.0, 5), ("11-20", 10.0, 5)]
    out = tmp_path / "curve.png"
    bleu_curve(rows, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_learning_curves_saves_png(tmp_path):
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("epoch,update,train_nll,val_nll\n1,100,5.0,5.5\n2,200,4.0,4.5\n")
    out = tmp_path / "learn.png"
    learning_curves(csv_path, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_learning_curves_skips_bad_rows(tmp_path):
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("epoch,update,train_nll,val_nll\nbad,100,5.0,5.5\n2,200,4.0,4.5\n")
    out = tmp_path / "learn.png"
    learning_curves(csv_path, out)
    assert out.exists()