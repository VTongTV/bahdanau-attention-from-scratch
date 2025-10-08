"""vocabulary statistics tests."""

from nmt.vocab.statistics import (
    coverage,
    oov_rate,
    token_freqs,
    type_token_ratio,
)


def test_token_freqs_counts_running_tokens():
    freqs = token_freqs([["a", "b"], ["a", "a", "c"]])
    assert freqs["a"] == 3
    assert freqs["b"] == 1
    assert freqs["c"] == 1


def test_type_token_ratio():
    freqs = token_freqs([["a", "a", "b"], ["b", "c"]])
    assert abs(type_token_ratio(freqs) - 3 / 5) < 1e-9


def test_type_token_ratio_empty():
    assert type_token_ratio(token_freqs([])) == 0.0


def test_coverage_share_of_running_tokens():
    freqs = token_freqs([["a", "b", "c"], ["a", "x"]])
    known = coverage(freqs, {"a", "b", "c"})
    assert abs(known - 4 / 5) < 1e-9


def test_oov_rate_complements_coverage():
    freqs = token_freqs([["a", "b"], ["x", "y"]])
    vocab = {"a", "b"}
    assert abs(oov_rate(freqs, vocab) + coverage(freqs, vocab) - 1.0) < 1e-9


def test_coverage_empty():
    assert coverage(token_freqs([]), {"a"}) == 0.0