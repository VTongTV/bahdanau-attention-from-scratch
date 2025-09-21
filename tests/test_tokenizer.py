"""tokenizer tests with known strings and apostrophes."""

from nmt.config import UNK
from nmt.vocab.tokenizer import tokenize, tokenize_line
from nmt.vocab.detokenizer import detokenize
from nmt.vocab.utf8 import normalize, fallback_unk
from nmt.vocab.vocabulary import Vocab


def test_punctuation_split():
    assert tokenize("Hello, world!") == ["Hello", ",", "world", "!"]


def test_apostrophe_kept():
    assert tokenize("It's a test") == ["It's", "a", "test"]
    assert tokenize("l'homme est ici") == ["l'homme", "est", "ici"]


def test_detokenize_roundtrip():
    tokens = tokenize("Hello, world! It's a test.")
    assert detokenize(tokens) == "Hello, world! It's a test."


def test_utf8_normalize():
    assert normalize("  a\t b  ") == "a b"


def test_unk_fallback():
    v = Vocab(5)
    v.count([["a", "b", "c"]])
    v.build()
    assert fallback_unk(["a", "zzz", "b"], v) == ["a", UNK, "b"]
    assert tokenize_line("a zzz b", v) == ["a", UNK, "b"]