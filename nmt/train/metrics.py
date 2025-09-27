"""nll metric tracking for train and validation runs."""


class Metrics:
    """running nll averages plus update counters."""

    def __init__(self):
        self.train_nll = 0.0
        self.val_nll = 0.0
        self.updates = 0
        self.batches = 0

    def update_train(self, nll: float) -> None:
        """fold one minibatch nll into the train average."""
        self.train_nll = (self.train_nll * self.batches + nll) / (self.batches + 1)
        self.batches += 1
        self.updates += 1

    def update_val(self, nll: float) -> None:
        """record the dev nll from a validation pass."""
        self.val_nll = nll

    def snapshot(self) -> dict:
        """a plain dict for csv logging."""
        return {
            "train_nll": self.train_nll,
            "val_nll": self.val_nll,
            "updates": self.updates,
            "batches": self.batches,
        }