import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

# noinspection PyUnresolvedReferences
from pose.args import args
from pose.data import split_train_dataset, get_autsl, PoseClassificationMockDataset, ZeroPadCollator
from pose.model import PoseSequenceClassification

if __name__ == '__main__':
  wandb_logger = WandbLogger(project="autsl", offline=False) # log_model=True,
  if wandb_logger.experiment.sweep_id is None:
    wandb_logger.log_hyperparams(args)

  # train = val = get_autsl('train[:1%]')
  train, val = split_train_dataset(get_autsl('train'), args.val_signers)

  # train = val = PoseClassificationMockDataset()

  collator = ZeroPadCollator()
  train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collator.collate)
  val_loader = DataLoader(val, batch_size=args.batch_size, collate_fn=collator.collate)

  model = PoseSequenceClassification()
  # model = PoseSequenceClassification.load_from_checkpoint(
  #   "/home/nlp/amit/sign-language/sign-language-recognition/pose/wandb/run-20210206_170707-111wpn4k/files/autsl/111wpn4k/checkpoints/epoch=26-step=606.ckpt")

  trainer = pl.Trainer(
    max_epochs=40,
    logger=wandb_logger,
    gpus=args.gpus)

  trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
  trainer.test(model, val_loader)
