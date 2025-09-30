"""run matrix configs for the paper comparison."""

from nmt.config import ExperimentConfig

# small-scale cpu dims. the ratios follow the paper architecture.
HIDDEN = 128
EMBEDDING = 64
ALIGNMENT_HIDDEN = 128
MAXOUT = 64
VOCAB = 3000


def rnnsearch_30():
    """rnnsearch trained on pairs up to 30 words."""
    return ExperimentConfig(
        model="rnnsearch",
        max_len=30,
        hidden=HIDDEN,
        embedding=EMBEDDING,
        alignment_hidden=ALIGNMENT_HIDDEN,
        maxout=MAXOUT,
        vocab_size=VOCAB,
    )