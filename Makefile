PYTHON := python

.PHONY: prepare train decode evaluate test lint

prepare:
	$(PYTHON) -c "from nmt.config import ExperimentConfig; from nmt.data.prepare import prepare; prepare(ExperimentConfig(max_len=30, vocab_size=3000, data_dir='data/wmt14'), 'experiments/runs/data30'); prepare(ExperimentConfig(max_len=50, vocab_size=3000, data_dir='data/wmt14'), 'experiments/runs/data50')"

train:
	$(PYTHON) nmt/train/train.py --data-dir experiments/runs/data30 --run-dir experiments/runs/$${RUN:-run} --model rnnsearch --max-len 30 --hidden 128 --embedding 64 --alignment-hidden 128 --maxout 64 --vocab-size 3000 --epochs 10 --patience 3 --seed 1

decode:
	$(PYTHON) -c "from nmt.exp.runner import decode_run; decode_run('experiments/runs/$${RUN:-run}', 'experiments/runs/data30')"

evaluate:
	$(PYTHON) -c "from nmt.exp.runner import collect_run; print(collect_run('experiments/runs/$${RUN:-run}', 'experiments/runs/data30', 'experiments/runs/data30/vocab.tgt'))"

test:
	$(PYTHON) -m pytest tests/ -q

lint:
	$(PYTHON) -m compileall -q nmt