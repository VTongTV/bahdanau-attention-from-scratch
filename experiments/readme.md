# experiments

how to reproduce the paper comparison on a small scale.

## the matrix

four runs: rnnsearch and rnnencdec, each in the 30-word and the 50-word
mode. all runs use the small cpu dims from `nmt/exp/configs.py`
(hidden 128, embedding 64, maxout 64, vocab 3000). the ratios follow
the paper architecture.

## prepare the data

```bash
python nmt/data/prepare.py --out experiments/runs/data30 --max-len 30
python nmt/data/prepare.py --out experiments/runs/data50 --max-len 50
```

## train

```bash
python nmt/train/train.py --data-dir experiments/runs/data30 \
  --run-dir experiments/runs/encdec30 --model rnnencdec --max-len 30 \
  --hidden 128 --embedding 64 --alignment-hidden 128 --maxout 64 \
  --vocab-size 3000 --epochs 10 --patience 3 --seed 1
```

or let the driver run the whole matrix:

```bash
python -c "from nmt.exp.runner import run_matrix; from nmt.exp.configs import rnnsearch_30, rnnencdec_30; run_matrix([rnnsearch_30(), rnnencdec_30()], 'experiments/runs')"
```

## decode and score a run

```bash
python -c "from nmt.exp.runner import decode_run; decode_run('experiments/runs/encdec30', 'experiments/runs/data30')"
```

this writes `test.npz.out` with one hypothesis per line (ids). the numbers
for the report tables then come from `nmt.eval.results.load_run_rows`.

## artifacts per run

- `train.csv` epoch, update, train nll, dev nll
- `checkpoint.best.pt` the best-dev model
- `train.log` and `train.err` the run console
- `test.npz.out` beam-decoded hypotheses of the test set

runs live under `experiments/runs/` and are gitignored.