"""paper hyperparameters and run configuration."""

from dataclasses import dataclass, fields, asdict


# model architecture (appendix a.2)
HIDDEN_SIZE = 1000
EMBEDDING_SIZE = 620
ALIGNMENT_HIDDEN = 1000
MAXOUT_SIZE = 500
VOCAB_SIZE = 30000
MAX_LEN_30 = 30
MAX_LEN_50 = 50

# training hyperparameters
OPTIMIZER = "sgd"
ADADELTA_RHO = 0.95
ADADELTA_EPS = 1e-6
MINI_BATCH = 80
GRAD_CLIP_NORM = 1.0
REBUCKET_EVERY = 20
REBUCKET_POOL = 1600

# decoding hyperparameters
BEAM_SIZE = 10
UNK_SUPPRESS = True

# test-mode subset sizes
TEST_MODE_TRAIN = 640
TEST_MODE_EVAL = 128

# initialization (appendix b.1)
ALIGNMENT_INIT_STD = 0.001
WEIGHT_INIT_STD = 0.01

# special tokens
BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"


@dataclass
class ExperimentConfig:
    """run configuration. overrides paper defaults for cpu runs."""

    model: str = "rnnsearch"
    max_len: int = MAX_LEN_30
    hidden: int = HIDDEN_SIZE
    embedding: int = EMBEDDING_SIZE
    alignment_hidden: int = ALIGNMENT_HIDDEN
    maxout: int = MAXOUT_SIZE
    vocab_size: int = VOCAB_SIZE
    minibatch: int = MINI_BATCH
    grad_clip: float = GRAD_CLIP_NORM
    adadelta_rho: float = ADADELTA_RHO
    adadelta_eps: float = ADADELTA_EPS
    rebucket_every: int = REBUCKET_EVERY
    rebucket_pool: int = REBUCKET_POOL
    beam_size: int = BEAM_SIZE
    unk_suppress: bool = UNK_SUPPRESS
    seed: int = 1
    epochs: int = 1
    max_updates: int = 0
    data_dir: str = "data/wmt14"
    run_dir: str = "experiments/runs"
    device: str = "auto"
    dtype: str = "float32"
    log_every: int = 10
    eval_every: int = 100
    patience: int = 3
    resume: str = ""
    test_mode: bool = False

    def to_dict(self) -> dict:
        """return the config as a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """build a config from a plain dict."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})