"""bucketed minibatch builder per the paper recipe."""

import numpy as np


def sort_by_length(pool, lengths):
    """order pool indices by sentence length."""
    order = np.argsort(lengths[pool].astype(np.int64), kind="stable")
    return [pool[i] for i in order]


def cut_batches(ordered, batch_size):
    """split an ordered index list into fixed-size batches."""
    return [ordered[i:i + batch_size] for i in range(0, len(ordered), batch_size)]


def rebucket(pool, lengths, batch_size=80):
    """sort a pool by length and cut it into batches."""
    return cut_batches(sort_by_length(pool, lengths), batch_size)


def fetch_pool(indices, start, pool_size):
    """pull the next pool of indices and return the next start."""
    end = min(start + pool_size, len(indices))
    return indices[start:end], end