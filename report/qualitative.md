# qualitative notes on alignment behavior

open log for what the trained rnnsearch models show on real samples.

## what to record

- monotonic diagonal mass per sentence class (short vs long sources)
- reordered pairs: adjective-noun flips, v2 word order in german-style pairs
- the paper's euro example: the alignment that stays monotone while the
  future tense moves to the end of the target sentence
- unk behavior: does the model emit unk on out-of-shortlist words
- length extremes: 1-2 word sentences and near-30 word sentences

## work in progress

trained models so far: rnnsearch-30 and rnnencdec-30 (small-scale dims).

| item | status |
| --- | --- |
| heatmaps on no-unk samples | pending decode of the test set |
| monotonicity stats over the test set | pending |
| reorder examples rank | pending |
| long-sentence comparison search vs baseline | pending |

## reference example from the paper (section 3.2)

the alignment of an english-french pair stays monotone except the future
tense, which aligns to the end of the french sentence. this is the case
our small models should reproduce in miniature on the 30-word subset.