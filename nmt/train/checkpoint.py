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
    model.load_state_dict(data["model"])
    if optimizer is not None:
        optimizer.load_state_dict(data["optimizer"])
    saved = ExperimentConfig.from_dict(data["config"])
    model.config = config
    return data["update"], saved