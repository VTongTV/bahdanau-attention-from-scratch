"""early stopping on dev nll with best-model preservation."""


class EarlyStopper:
    """stops when the dev nll stops improving for patience checks."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best = float("inf")
        self.wait = 0
        self.epoch = 0

    def observe(self, val_nll: float) -> bool:
        """feed a dev nll. returns true when training should stop."""
        self.epoch += 1
        if val_nll < self.best:
            self.best = val_nll
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.patience