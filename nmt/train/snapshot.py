"""keep the best-dev checkpoint for a run."""

from pathlib import Path


class BestSnapshot:
    """compare dev nll and save the best model state so far."""

    def __init__(self, path, save_fn, best=float("inf")):
        self.path = Path(path)
        self.save_fn = save_fn
        self.best = best
        self.best_update = 0

    def observe(self, dev_nll, update):
        """save on improvement and report whether dev nll improved."""
        if dev_nll < self.best:
            self.best = dev_nll
            self.best_update = update
            self.save_fn(self.path)
            return True
        return False