# neural machine translation by jointly learning to align and translate

a from-scratch implementation of bahdanau, cho & bengio (iclr 2015, arXiv:1409.0473v7).

## what this is

the paper proposes rnnsearch: a bidirectional gru encoder with a soft
attention decoder. each target word gets its own context vector, a weighted
sum of source annotations. the baseline rnnencdec (cho 2014) keeps the same
code with the context fixed to the last forward state.

## models

| model | description |
|-------|-------------|
| rnnsearch | bidirectional encoder, attention decoder, maxout deep output |
| rnnencdec | same code, context fixed to the last forward state |

each model trains in two modes: pairs up to 30 words and pairs up to 50 words.

## layout

- `nmt/config.py` paper constants (hidden 1000, embedding 620, maxout 500)
- `nmt/model/` gru cells, encoder, alignment, attention, decoder, head
- `nmt/data/` parallel corpus, filtering, bucketing, collation
- `nmt/train/` adadelta, gradient clip, checkpoints, early stop
- `nmt/decode/` greedy and beam search with unk guard
- `nmt/eval/` bleu from scratch, subsets, tables
- `nmt/exp/` run configs and experiment driver

## roadmap

1. foundation: scaffold, config, vocab, tokenizer
2. data pipeline: filters, splits, buckets, collator
3. model core: gru, birnn, alignment, attention, decoder, head
4. training: adadelta, clipping, checkpoints, early stop
5. decoding: greedy, beam, bleu
6. experiments: rnnsearch vs rnnencdec, 30/50 modes
7. analysis: alignments, monotonicity, length curves
8. report and release

## usage

```bash
python -m nmt.cli train --config experiments.yaml
python -m nmt.cli translate --checkpoint runs/best.pt --input test.en
python -m nmt.cli evaluate --hypotheses out.fr --references test.fr
```

## tests

```bash
python -m pytest tests/ -v
```