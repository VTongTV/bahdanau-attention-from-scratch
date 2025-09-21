"""vocab tests for shortlist cut, unk mapping and save/load."""

from pathlib import Path

from nmt.config import BOS, EOS, UNK
from nmt.vocab.vocabulary import Vocab
from nmt.vocab.special import special_ids, is_special


def make_vocab(size=10):
    v = Vocab(size)
    v.count([["a", "b", "c", "a", "a"], ["b", "b", "d"], ["e", "f", "g", "h", "i", "j", "k"]])
    v.build()
    return v


def test_shortlist_cut():
    v = make_vocab(10)
    assert len(v) == 10
    assert v.token[0] == BOS
    assert v.token[1] == EOS
    assert v.token[2] == UNK


def test_unk_mapping():
    v = make_vocab(10)
    unk_id = v.stoi[UNK]
    assert v.id("zzz_not_in_vocab") == unk_id
    assert v.id("a") != unk_id


def test_special_ids_stable():
    v = make_vocab(10)
    ids = special_ids(v)
    assert ids["bos"] == 0
    assert ids["eos"] == 1
    assert ids["unk"] == 2
    assert is_special(BOS) and is_special(EOS) and is_special(UNK)
    assert not is_special("a")


def test_save_load(tmp_path: Path):
    v = make_vocab(10)
    path = tmp_path / "vocab.txt"
    v.save(path)
    v2 = Vocab.load(path)
    assert v2.token == v.token
    assert v2.id("a") == v.id("a")
    assert v2.id("zzz") == v2.stoi[UNK]