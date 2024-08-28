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
        self.train_accuracy = Accuracy(num_classes=226, task="multiclass")
        self.train_loss = MeanMetric()
        self.val_accuracy = Accuracy(num_classes=226, task="multiclass")
        self.val_loss = MeanMetric()
        self.test_accuracy = Accuracy(num_classes=226, task="multiclass")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def pred(self, *args):
        y_hat, signer_hat = self(*args)
        return torch.argmax(y_hat, dim=1)

    def step(self, split: str, batch, batch_idx):
        y = batch["label"]
        signer = batch["signer"]
        y_hat, signer_hat = self(batch)
        
        loss = F.cross_entropy(y_hat, y)
        if split == "training" and self.current_epoch > self.signer_loss_patience and self.signer_loss != 0:
            loss += self.signer_loss * F.cross_entropy(signer_hat, signer)

        # Using metrics for training and validation
        if split == 'training':
            self.train_accuracy(y_hat, y)
            self.train_loss(loss)
            self.log(f'{split}_acc', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["pose"].size(0))
            self.log(f'{split}_loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["pose"].size(0))
        elif split == 'validation':
            self.val_accuracy(y_hat, y)
            self.val_loss(loss)
            self.log(f'{split}_acc', self.val_accuracy, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["pose"].size(0))
            self.log(f'{split}_loss', self.val_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["pose"].size(0))
        return {"loss": loss, "pred": torch.argmax(y_hat, dim=1), "target": y}

    def on_train_epoch_end(self):
        self.log('train_acc_epoch', self.train_accuracy.compute())
        self.log('train_loss_epoch', self.train_loss.compute())
        self.train_accuracy.reset()
        self.train_loss.reset()

    def on_validation_epoch_end(self):
        self.log('val_acc_epoch', self.val_accuracy.compute())
        self.log('val_loss_epoch', self.val_loss.compute())
        self.val_accuracy.reset()
        self.val_loss.reset()

    def on_test_epoch_end(self):
        self.log('test_acc_epoch', self.test_accuracy.compute())
        self.test_accuracy.reset()

    def training_step(self, batch, batch_idx):
        return self.step("training", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step("validation", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.step("test", batch, batch_idx)
