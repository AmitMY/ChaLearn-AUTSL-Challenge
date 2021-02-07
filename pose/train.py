import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

# noinspection PyUnresolvedReferences
from pose.data import PoseClassificationDataset, split_train_dataset, get_autsl
from pose.model import PoseSequenceClassification

wandb_logger = WandbLogger(project="autsl", log_model=True, offline=True)

train = get_autsl('train')
train, val = split_train_dataset(train, {0, 12, 21, 22})

gpus = 1
batch_size = max(1, gpus) * 1024
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size)

model = PoseSequenceClassification()
# model = PoseSequenceClassification.load_from_checkpoint(
#   "/home/nlp/amit/sign-language/sign-language-recognition/pose/wandb/run-20210206_170707-111wpn4k/files/autsl/111wpn4k/checkpoints/epoch=26-step=606.ckpt")

trainer = pl.Trainer(
  max_epochs=100,
  logger=wandb_logger,
  gpus=gpus)

trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
trainer.test(model, val_loader)
