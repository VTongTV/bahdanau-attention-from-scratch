"""dev set assembly and test split from the wmt14 files."""

import re
from pathlib import Path

_SEG = re.compile(r"<seg id=\"\d+\">(.*?)</seg>")


def extract_sgm(path):
    """yield the text of each seg element in an sgm file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _SEG.search(line)
            if m:
                yield m.group(1)


def dev_pairs(data_dir):
    """yield pairs from news-test-2012 plus news-test-2013."""
    d = Path(data_dir) / "dev"
    for year in ("newstest2012", "newstest2013"):
        with open(d / f"{year}.en", encoding="utf-8") as fs, open(d / f"{year}.fr", encoding="utf-8") as ft:
            for s, t in zip(fs, ft):
                yield s.rstrip("\n"), t.rstrip("\n")


def test_pairs(data_dir):
    """yield pairs from news-test-2014, reading the sgm segments."""
    d = Path(data_dir) / "test"
    en = list(extract_sgm(d / "newstest2014-fren-src.en.sgm"))
    fr = list(extract_sgm(d / "newstest2014-fren-src.fr.sgm"))
    for s, t in zip(en, fr):
        yield s, t