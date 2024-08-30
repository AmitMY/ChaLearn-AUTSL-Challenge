from torch import nn
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

        self.metrics = nn.ModuleDict()
        for split in ["training", "validation", "test"]:
            self.metrics[f"acc_{split}"] = Accuracy(num_classes=226, task="multiclass")
            if split != "test":
                self.metrics[f"mean_loss_{split}"] = MeanMetric()

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

        self.metrics[f"acc_{split}"](y_hat, y)
        self.metrics[f"mean_loss_{split}"](loss)
        self.log(f'{split}_acc', self.metrics[f"acc_{split}"].compute(), on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["pose"].size(0))
        self.log(f'{split}_loss', self.metrics[f"mean_loss_{split}"].compute(), on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["pose"].size(0))

        return {"loss": loss, "pred": torch.argmax(y_hat, dim=1), "target": y}

    def on_epoch_end(self):
        for split in ["training", "validation", "test"]:
            self.log(f'{split}_acc_epoch', self.metrics[f"acc_{split}"].compute())
            self.metrics[f"acc_{split}"].reset()
            if split != "test":
                self.log(f'{split}_loss_epoch', self.metrics[f"mean_loss_{split}"].compute())
                self.metrics[f"mean_loss_{split}"].reset()

    def training_step(self, batch, batch_idx):
        return self.step("training", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step("validation", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.step("test", batch, batch_idx)
