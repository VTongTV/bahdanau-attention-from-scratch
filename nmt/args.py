"""cli flags translated into an ExperimentConfig."""

import argparse

from nmt.config import ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    """return the cli parser with all run flags."""
    p = argparse.ArgumentParser(description="train or evaluate an nmt model")
    p.add_argument("--model", choices=["rnnsearch", "rnnencdec"], default="rnnsearch")
    p.add_argument("--max-len", type=int, default=30)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--embedding", type=int, default=None)
    p.add_argument("--alignment-hidden", type=int, default=None)
    p.add_argument("--maxout", type=int, default=None)
    p.add_argument("--vocab-size", type=int, default=None)
    p.add_argument("--minibatch", type=int, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--adadelta-rho", type=float, default=None)
    p.add_argument("--adadelta-eps", type=float, default=None)
    p.add_argument("--rebucket-every", type=int, default=None)
    p.add_argument("--rebucket-pool", type=int, default=None)
    p.add_argument("--beam-size", type=int, default=None)
    p.add_argument("--no-unk-suppress", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-updates", type=int, default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--test-mode", action="store_true")
    return p


def config_from_args(argv=None) -> ExperimentConfig:
    """parse argv and overlay non-default flags on the base config."""
    args = build_parser().parse_args(argv)
    base = ExperimentConfig()
    overrides = {}
    for name, value in vars(args).items():
        if value is None:
            continue
        key = name.replace("-", "_")
        if key == "no_unk_suppress":
            overrides["unk_suppress"] = not value
        elif key in base.to_dict():
            overrides[key] = value
    return ExperimentConfig(**{**base.to_dict(), **overrides})