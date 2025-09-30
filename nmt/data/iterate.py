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
    """yield padded batches. re-bucket every pool and carry the tail."""
    start = 0
    carry = np.asarray([], dtype=np.int64)
    while start < len(order):
        pool, start = fetch_pool(order, start, pool_size)
        batches = rebucket(np.concatenate([carry, pool]), store.src_len, batch_size)
        carry = np.asarray(batches.pop(), dtype=np.int64)
        for batch in batches:
            yield _collate_rows(store, batch)
    if len(carry):
        yield _collate_rows(store, carry)


def _collate_rows(store, batch):
    """pad one index batch into collated tensors."""
    src_rows = [store.src_row(i) for i in batch]
    tgt_rows = [store.tgt_row(i) for i in batch]
    return collate(src_rows, tgt_rows)