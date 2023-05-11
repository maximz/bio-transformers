from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torchmetrics
from torch.nn import functional as F  # noqa: N812 pylint: disable=wrong-import-order

from .optimizer import lr_update


class LightningModule(pl.LightningModule):
    """Create lightning model to use ddp"""

    def __init__(
        self,
        model,
        alphabet,
        vocab_size: int,
        lr: float,
        warmup_end_lr: float,
        warmup_updates: int = 10,
        warmup_init_lr: float = 1e-7,
    ):
        super().__init__()
        self.model = model
        self.alphabet = alphabet
        self.vocab_size = vocab_size
        self.lr = lr
        self.automatic_optimization = True
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = min(warmup_init_lr, lr)
        self.lr_step = (warmup_end_lr - self.warmup_init_lr) / warmup_updates
        self.decay_factor = warmup_end_lr * warmup_updates ** 0.5
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=vocab_size)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=vocab_size)

    def forward(self, x):
        return self.model(x)["logits"]

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict]]:
        """Configure the optimizer and learning rate scheduler.

        Returns:
            - list of optimizers.
            - list of lr schedulers.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda x: lr_update(
                    num_updates=x,
                    warmup_updates=self.warmup_updates,
                    warmup_init_lr=self.warmup_init_lr,
                    lr_step=self.lr_step,
                    decay_factor=self.decay_factor,
                ),
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [lr_scheduler]

    def cross_entropy_loss(self, logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        cross_entropy_per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            # reshape(-1) flattens out the 2D tensor into a 1D tensor
            targets.reshape(-1),
            # set reduction="none" to get loss for each instance,
            # then do weighted sum ourselves
            # reduction="sum",
            reduction="none",
            ignore_index=self.alphabet.padding_idx,
        )
        return (cross_entropy_per_token * weights.reshape(-1)).sum()

    def training_step(self, train_batch, batch_idx):
        # train_batch is a set of 2D tensors of shape #sequences x #tokens
        # in each tensor: each row is a sequence; each entry in the row corresponds to a particular token.
        tokens, target, weights = train_batch
        logits = self.forward(tokens)
        loss = self.cross_entropy_loss(logits, target, weights)

        masked_preds, masked_targets = self.get_tensor_accuracy(logits, target)
        # TODO: incorporate weights (not supported natively by torchmetrics.Accuracy)
        self.train_acc(masked_preds, masked_targets)

        masked_tokens = target.ne(self.alphabet.padding_idx)
        sample_size = masked_tokens.int().sum()
        loss = loss / sample_size

        self.log_dict(
            {"train_loss": loss, "train_acc": self.train_acc},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, val_batch, batch_idx):
        """Log the loss and metrics for a batch.

        Args:
            batch: batch input.
            batch_idx: index of the batch.
        """
        # val_batch is a set of 2D tensors of shape #sequences x #tokens
        # in each tensor: each row is a sequence; each entry in the row corresponds to a particular token.
        tokens, target, weights = val_batch
        logits = self.forward(tokens)
        loss = self.cross_entropy_loss(logits, target, weights)

        masked_preds, masked_targets = self.get_tensor_accuracy(logits, target)
        # TODO: incorporate weights (not supported natively by torchmetrics.Accuracy)
        self.val_acc(masked_preds, masked_targets)

        masked_tokens = target.ne(self.alphabet.padding_idx)
        sample_size = masked_tokens.int().sum()
        loss = loss / sample_size

        self.log_dict(
            {"val_loss": loss, "val_acc": self.val_acc},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def get_tensor_accuracy(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate accuracy for multi-masking, summed over batch.

        Args:
            logits: prediction from the model, shape = (batch, len_tokens, len_vocab)
            targets: ground truth, shape = (batch, len_tokens)

        Returns:
            accuracy value.
        """
        preds = torch.argmax(logits, dim=-1)  # (batch, len_tokens)
        masked_tokens = targets.ne(self.alphabet.padding_idx)

        masked_preds = torch.masked_select(preds, masked_tokens)
        masked_targets = torch.masked_select(targets, masked_tokens)

        return masked_preds.detach().cpu(), masked_targets.detach().cpu()
