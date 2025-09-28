"""trainer loop with minibatch iteration and loss logging."""

import torch

from nmt.config import ExperimentConfig
from nmt.data.iterate import epoch_batches, shuffled_order
from nmt.train.clip import clip_gradients
from nmt.train.loss import masked_nll


class Trainer:
    """walks minibatches, updates parameters, tracks the loss."""

    def __init__(self, model, optimizer, config: ExperimentConfig):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.updates = 0

    def train_step(self, batch):
        """one sgd update over a padded minibatch."""
        src, src_mask, tgt, tgt_mask = batch
        self.optimizer.zero_grad()
        logits = self.model(src, tgt, src_mask)
        loss = masked_nll(logits, tgt, tgt_mask)
        loss.backward()
        clip_gradients(self.model, self.config.grad_clip)
        self.optimizer.step()
        self.updates += 1
        return loss.item()

    def run_epoch(self, store, epoch, log_every: int):
        """one sequential pass over the shuffled sentences."""
        order = shuffled_order(len(store), self.config.seed + epoch)
        losses = []
        for batch in epoch_batches(
            store,
            order,
            self.config.minibatch,
            self.config.rebucket_pool,
        ):
            loss = self.train_step(batch)
            losses.append(loss)
            if self.updates % log_every == 0:
                print(f"step {self.updates} loss {loss:.4f}", flush=True)
        return sum(losses) / len(losses)

    def validate(self, store, order):
        """dev nll in no-grad mode over the given sentences."""
        self.model.eval()
        total = 0.0
        seen = 0
        for batch in epoch_batches(
            store,
            order,
            self.config.minibatch,
            self.config.rebucket_pool,
        ):
            src, src_mask, tgt, tgt_mask = batch
            with torch.no_grad():
                logits = self.model(src, tgt, src_mask)
            total += masked_nll(logits, tgt, tgt_mask).item() * src.shape[0]
            seen += src.shape[0]
        self.model.train()
        return total / max(seen, 1)