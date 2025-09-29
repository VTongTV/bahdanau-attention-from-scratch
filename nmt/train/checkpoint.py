"""checkpoint save and load for model optimizer and config."""

import json
from pathlib import Path

import torch

from nmt.config import ExperimentConfig


def save_checkpoint(path, model, optimizer, config: ExperimentConfig, update: int):
    """write model optimizer config and update count to one file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.to_dict(),
            "update": update,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, config: ExperimentConfig):
    """restore a training state from a checkpoint file."""
    data = torch.load(path, weights_only=False)
    saved = ExperimentConfig.from_dict(data["config"])
    if saved.model != config.model or saved.hidden != config.hidden:
        raise ValueError("checkpoint architecture does not match the cli config")
    if saved.vocab_size != config.vocab_size:
        raise ValueError("checkpoint vocabulary does not match the cli config")
    model.load_state_dict(data["model"])
    if optimizer is not None:
        optimizer.load_state_dict(data["optimizer"])
    model.config = config
    return data["update"], saved