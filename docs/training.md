# training

how a run follows the paper recipe (section 3.2 and appendix a.3).

## the recipe

- sgd with adadelta (rho 0.95, eps 1e-6) from `nmt/train/optimizer.py`.
- gradients clip at l2 norm 1.0 from `nmt/train/clip.py`.
- minibatches of 80 sentences from `nmt/config.py`.
- the data shuffles once per epoch and walks sequentially.
- every pool of 1600 sentences sorts by length and cuts into 20
  minibatches of 80. that is the re-bucketing cadence from the paper.
- the loss is the mean nll over the sentences in the batch, where each
  sentence nll averages over its non-pad words (`nmt/train/loss.py`).
- dev nll runs at every epoch boundary in no-grad mode.
- training stops when the dev nll plateaus (patience from the config).
- the best dev model gets its own checkpoint file.

## running

the entry point is `python -m nmt.train.train` (or the cli in
`nmt/cli.py`). flags map to the config in `nmt/args.py`.

```
python -m nmt.train.train --model rnnsearch-30 --data-dir data/wmt14 \
    --run-dir experiments/runs/rs30 --epochs 1
```

cpu runs shrink hidden, embedding and vocab. that is a run config,
not an architecture change. the paper constants stay in `nmt/config.py`.

## resume

`--resume experiments/runs/rs30/checkpoint.last.pt` restores the model
and optimizer state and continues from the saved update count. the
checkpoint stores the config too. a mismatch in model kind, hidden size
or vocab size raises before any weights load.

## run artifacts

each run directory holds:

| file | content |
| --- | --- |
| `train.csv` | one row per epoch: step, train nll, dev nll |
| `checkpoint.last.pt` | the last epoch state |
| `checkpoint.best.pt` | the best dev nll state |

## small-scale numbers

the paper trains for days on one gpu (table 2). our runs are cpu runs
of hours. the expected bleed table in `nmt/config.py` stays the target
for the same scale; a bleed more than 5 points under its band needs a
check with the human before continuing.