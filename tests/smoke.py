"""smoke test that the package imports and config loads."""

from nmt.config import ExperimentConfig


def test_config_defaults():
    c = ExperimentConfig()
    assert c.hidden == 1000
    assert c.embedding == 620
    assert c.maxout == 500
    assert c.vocab_size == 30000


def test_config_roundtrip():
    c = ExperimentConfig(model="rnnencdec", max_len=50)
    assert ExperimentConfig.from_dict(c.to_dict()) == c