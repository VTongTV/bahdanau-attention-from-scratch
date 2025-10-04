"""length statistics tests."""

from nmt.analysis.lengths import (
    length_histogram,
    length_stats,
    longest_rows,
    ratio_stats,
)


def test_length_histogram_buckets_cap():
    counts = length_histogram([2, 2, 9, 9, 9], max_len=3)
    assert counts == [0, 2, 3]


def test_length_stats_known():
    stats = length_stats([2, 4, 6])
    assert stats["mean"] == 4.0
    assert stats["median"] == 4.0
    assert stats["count"] == 3


def test_length_stats_empty():
    stats = length_stats([])
    assert stats["count"] == 0
    assert stats["mean"] == 0.0


def test_ratio_stats_when_empty():
    stats = ratio_stats([], [])
    assert stats["count"] == 0


def test_ratio_stats_known():
    stats = ratio_stats([2, 4], [4, 8])
    assert abs(stats["mean"] - 2.0) < 1e-9


def test_longest_rows_orders():
    class FakeStore:
        def __init__(self):
            self.rows = [[1, 2, 3], [1], [1, 2, 3, 4, 5]]

        def src_row(self, i):
            return self.rows[i]

        def __len__(self):
            return len(self.rows)

    assert longest_rows(FakeStore(), 2) == [2, 0]