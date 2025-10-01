"""bleu and subset evaluation tests on known pairs."""

import pytest

from nmt.eval.bleu import sentence_bleu
from nmt.eval.corpus_bleu import corpus_bleu
from nmt.eval.subsets import length_buckets, no_unk_pairs


def test_identical_sentence_scores_one():
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    assert sentence_bleu(tokens, tokens) == pytest.approx(1.0)


def test_disjoint_sentences_score_zero():
    pred = ["the", "cat"]
    ref = ["red", "dog"]
    assert sentence_bleu(pred, ref) == 0.0


def test_short_prediction_gets_brevity_penalty():
    pred = ["hello"]
    ref = ["hello", "world"]
    assert sentence_bleu(pred, ref) < 1.0


def test_empty_prediction_scores_zero():
    assert sentence_bleu([], ["a", "b"]) == 0.0


def test_corpus_bleu_matches_single_pair():
    preds = [["the", "cat", "sat"]]
    refs = [["the", "cat", "sat"]]
    assert corpus_bleu(preds, refs) == pytest.approx(sentence_bleu(preds[0], refs[0]))


def test_corpus_bleu_identical_corpora_score_one():
    preds = [["a", "b"], ["c", "d", "e"]]
    assert corpus_bleu(preds, preds) == pytest.approx(1.0)


def test_no_unk_pairs_filters_by_token():
    srcs = [["a", "<unk>"], ["b", "c"]]
    preds = [["x"], ["y"]]
    refs = [["u"], ["v", "w"]]
    keep = no_unk_pairs(srcs, preds, refs, "<unk>")
    assert len(keep) == 1
    assert keep[0][0] == ["b", "c"]


def test_length_buckets_cover_all_pairs():
    srcs = [["a"] * 5, ["b"] * 15, ["c"] * 35]
    preds = [["p"] * 4, ["q"] * 12, ["r"] * 30]
    refs = [["p"] * 4, ["q"] * 12, ["r"] * 30]
    rows = length_buckets(srcs, preds, refs)
    assert sum(count for _, _, count in rows) == 3
    assert [label for label, _, _ in rows] == sorted(label for label, _, _ in rows)