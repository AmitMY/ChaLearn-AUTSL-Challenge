from collections import Counter
from itertools import chain
from zipfile import ZipFile

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, MetricCollection, MeanMetric


class PLModule(pl.LightningModule):
    def __init__(self, sign_loss=1, signer_loss=1, signer_loss_patience=0):
        super().__init__()
        
        self.sign_loss = sign_loss
        self.signer_loss = signer_loss
        self.signer_loss_patience = signer_loss_patience

        # Metrics for training and validation
        self.metrics = {split: Accuracy(num_classes=226, task="multiclass") for split in ["training", "validation", "test"]}
        self.mean_loss = {split: MeanMetric() for split in ["training", "validation", "test"]}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def pred(self, *args):
        y_hat, signer_hat = self(*args)
        return torch.argmax(y_hat, dim=1)

    def step(self, split: str, batch, batch_idx):
        y = batch["label"]
        signer = batch["signer"]
        y_hat, signer_hat = self(batch)

        self.metrics[split] = self.metrics[split].to(y_hat.device) if self.metrics[split].device != y_hat.device else self.metrics[split]
        self.mean_loss[split] = self.mean_loss[split].to(y_hat.device) if self.mean_loss[split].device != y_hat.device else self.mean_loss[split]

        loss = F.cross_entropy(y_hat, y)
        if split == "training" and self.current_epoch > self.signer_loss_patience and self.signer_loss != 0:
            loss += self.signer_loss * F.cross_entropy(signer_hat, signer)

        # Using metrics for training and validation

        self.metrics[split](y_hat, y)
        self.mean_loss[split](loss)
        self.log(f'{split}_acc', self.metrics[split].compute(), on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["pose"].size(0))
        self.log(f'{split}_loss', self.mean_loss[split].compute(), on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["pose"].size(0))
        return {"loss": loss, "pred": torch.argmax(y_hat, dim=1), "target": y}

    def on_train_epoch_end(self):
        self.log('train_acc_epoch', self.metrics['training'].compute())
        self.log('train_loss_epoch', self.mean_loss['training'].compute())
        self.metrics['training'].reset()
        self.mean_loss['training'].reset()

    def on_validation_epoch_end(self):
        self.log('val_acc_epoch', self.metrics['validation'].compute())
        self.log('val_loss_epoch', self.mean_loss['validation'].compute())
        self.metrics['validation'].reset()
        self.mean_loss['validation'].reset()

    def on_test_epoch_end(self):
        self.log('test_acc_epoch', self.test_accuracy.compute())
        self.test_accuracy.reset()

    def training_step(self, batch, batch_idx):
        return self.step("training", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step("validation", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.step("test", batch, batch_idx)
