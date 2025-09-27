"""trainer loop with minibatch iteration and loss logging."""

import torch

from nmt.config import ExperimentConfig
from nmt.data.collate import collate
from nmt.data.iterate import shuffled_order
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
        for start in range(0, len(order), self.config.minibatch):
            rows = order[start:start + self.config.minibatch]
            src_rows = [store.src_row(i) for i in rows]
            tgt_rows = [store.tgt_row(i) for i in rows]
            loss = self.train_step(collate(src_rows, tgt_rows))
            losses.append(loss)
            if self.updates % log_every == 0:
                print(f"step {self.updates} loss {loss:.4f}", flush=True)
        return sum(losses) / len(losses)