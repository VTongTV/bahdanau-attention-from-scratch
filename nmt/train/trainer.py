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

    def _device(self):
        """the device the model parameters live on."""
        return next(self.model.parameters()).device

    def train_step(self, batch):
        """one sgd update over a padded minibatch."""
        src, src_mask, tgt, tgt_mask = batch
        device = self._device()
        src = src.to(device)
        tgt = tgt.to(device)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        self.optimizer.zero_grad()
        logits = self.model(src, tgt, src_mask)
        loss = masked_nll(logits, tgt[:, 1:], tgt_mask[:, 1:])
        loss.backward()
        self.last_gnorm = self._grad_norm()
        clip_gradients(self.model, self.config.grad_clip)
        self.optimizer.step()
        self.updates += 1
        return loss.item()

    def _grad_norm(self):
        """l2 norm summed over all parameter gradients."""
        sq = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                sq += float(param.grad.norm() ** 2)
        return sq ** 0.5

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
                rms_g, rms_dx, scale = self.optimizer.statistics()
                print(f"step {self.updates} loss {loss:.4f} "
                      f"gnorm {self.last_gnorm:.4f} "
                      f"rms_g {rms_g:.5f} rms_dx {rms_dx:.5f} scale {scale:.4f}",
                      flush=True)
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
            device = self._device()
            src = src.to(device)
            tgt = tgt.to(device)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)
            with torch.no_grad():
                logits = self.model(src, tgt, src_mask)
            total += masked_nll(logits, tgt[:, 1:], tgt_mask[:, 1:]).item() * src.shape[0]
            seen += src.shape[0]
        self.model.train()
        return total / max(seen, 1)