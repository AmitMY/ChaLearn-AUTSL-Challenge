from collections import Counter

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import metrics


class PLModule(pl.LightningModule):
    def __init__(self, sign_loss=1, signer_loss=1, signer_loss_patience=0):
        super().__init__()

        self.sign_loss = sign_loss
        self.signer_loss = signer_loss
        self.signer_loss_patience = signer_loss_patience

        self.metrics = {
            "training": metrics.Accuracy(),
            "validation": metrics.Accuracy(),
            "test": metrics.Accuracy(),
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def pred(self, *args):
        y_hat, signer_hat = self.forward(*args)
        return torch.argmax(y_hat, dim=1)

    # Define steps

    def step(self, split: str, batch, batch_idx):
        y = batch["label"]
        signer = batch["signer"]
        y_hat, signer_hat = self(batch)

        # self.logger.experiment.add_image('example_images', grid, 0)

        loss = F.cross_entropy(y_hat, y)
        if split == "training" and self.current_epoch > self.signer_loss_patience and self.signer_loss != 0:
            loss = self.sign_loss * loss + self.signer_loss * F.cross_entropy(signer_hat, signer)

        return {
            "loss": loss,
            "signer": signer,
            "pred": torch.argmax(y_hat, dim=1),
            "target": y
        }

    def step_end(self, split: str, outputs):
        self.log(f'{split}_loss', outputs["loss"], on_step=True, on_epoch=False)
        acc = self.metrics[split](outputs["pred"].cpu(), outputs["target"].cpu())
        self.log(f'{split}_acc', acc, on_step=True, on_epoch=False, prog_bar=True)
        return outputs

    def epoch_end(self, split: str, outputs):
        correct_by_signer = Counter()
        signer_count = Counter()

        loss = torch.stack([o["loss"] for o in outputs], dim=0).mean()
        pred = torch.cat([o["pred"] for o in outputs])
        target = torch.cat([o["target"] for o in outputs])
        signer = torch.cat([o["signer"] for o in outputs]).cpu().numpy()
        correct = (pred == target).cpu().numpy()
        for s, v in zip(signer, correct):
            signer_count[s] += 1
            if v:
                correct_by_signer[s] += 1

        print("\n\nAcc by signer:")
        print("Total %.3f" % (sum(correct_by_signer.values()) / sum(signer_count.values())))
        for s in sorted(signer_count.keys()):
            print("Signer\t%d\t%.3f\t(%d / %d)" %
                  (s, correct_by_signer[s] / signer_count[s], correct_by_signer[s], signer_count[s]))
        print("\n")

        self.log(f'{split}_loss_epoch', loss, on_step=False, on_epoch=True)
        self.log(f'{split}_acc_epoch', self.metrics[split].compute(), on_step=False, on_epoch=True)

    # Training steps

    def training_step(self, batch, batch_idx):
        return self.step("training", batch, batch_idx)

    def training_step_end(self, outputs):
        return self.step_end("training", outputs)

    def training_epoch_end(self, outputs):
        return self.epoch_end("training", outputs)

    # Validation steps

    def validation_step(self, batch, batch_idx):
        return self.step("validation", batch, batch_idx)

    def validation_step_end(self, outputs):
        return self.step_end("validation", outputs)

    def validation_epoch_end(self, outputs):
        return self.epoch_end("validation", outputs)

    # Validation steps

    def test_step(self, batch, batch_idx):
        return self.step("test", batch, batch_idx)

    def test_step_end(self, outputs):
        return self.step_end("test", outputs)

    def test_epoch_end(self, outputs):
        return self.epoch_end("test", outputs)
