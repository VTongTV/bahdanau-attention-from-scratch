# performance notes

small notes on runtime decisions and their cost.

## alignment precompute

the alignment score is

    e_ij = v_a^T tanh(w_a s_{i-1} + u_a h_j)

u_a h_j depends only on the source annotation, so it is computed once per
encoder run and reused for every decoder step. the cache is mathematically
identical to recomputing per step:

    reference: u_a @ annotations (every step)
    cached:    stored at encoder time (every step)

the unit test `test_cached_equals_direct` in `tests/test_attention.py`
checks both paths agree to floating point. the cache saves the matrix
product for every decoder step at the cost of one extra product per source
sentence.

the same trick applies at decode time: attention scores are evaluated
ty * tx times per pair in the worst case, and the cached encodings cut
that to ty + tx products.

## decode cost

beam search of width 10 evaluates the head on 10 hypotheses per step.
the unk guard masks one logit before the softmax, which is a constant
cost per step.

## measured numbers

to be filled from the run logs:
- sentences/sec on the gpu for the 30-word and 50-word corpora
- seconds per test-set decode (3003 sentences, beam 10)
- rms_g / rms_dx evolution over training (adadelta adaptation)