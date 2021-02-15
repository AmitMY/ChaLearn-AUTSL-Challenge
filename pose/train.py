import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

# noinspection PyUnresolvedReferences
from pose.args import args
from pose.data import split_train_dataset, get_autsl
from pose.model import PoseSequenceClassification

wandb_logger = WandbLogger(project="autsl", log_model=True, offline=False)
if wandb_logger.experiment.sweep_id is None:
  wandb_logger.log_hyperparams(args)

# train = get_autsl('train[:1%]')
train = get_autsl('train')
train, val = split_train_dataset(train, args.val_signers)

train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=args.batch_size)

model = PoseSequenceClassification()
# model = PoseSequenceClassification.load_from_checkpoint(
#   "/home/nlp/amit/sign-language/sign-language-recognition/pose/wandb/run-20210206_170707-111wpn4k/files/autsl/111wpn4k/checkpoints/epoch=26-step=606.ckpt")

trainer = pl.Trainer(
  max_epochs=20,
  logger=wandb_logger,
  gpus=args.gpus)

trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
trainer.test(model, val_loader)
