"""deterministic minibatch iteration with a seeded shuffle."""

import numpy as np

from nmt.data.bucket import fetch_pool, rebucket
from nmt.data.collate import collate


def shuffled_order(n, seed):
    """return a seeded permutation of the sentence indices."""
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    rng.shuffle(order)
    return order


def epoch_batches(store, order, batch_size=80, pool_size=1600):
    """yield padded batches. re-bucket every pool as the paper does."""
    start = 0
    while start < len(order):
        pool, start = fetch_pool(order, start, pool_size)
        for batch in rebucket(pool, store.src_len, batch_size):
            src_rows = [store.src_row(i) for i in batch]
            tgt_rows = [store.tgt_row(i) for i in batch]
            yield collate(src_rows, tgt_rows)