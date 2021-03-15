import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

# noinspection PyUnresolvedReferences
from .args import args
from .data import get_autsl, ZeroPadCollator
from .model import PoseSequenceClassification

if __name__ == '__main__':
    if not args.no_wandb:
        logger = WandbLogger(project="autsl", log_model=False, offline=False)
        if logger.experiment.sweep_id is None:
            logger.log_hyperparams(args)
    else:
        logger = None

    collator = ZeroPadCollator()

    train = get_autsl('train')
    train.is_train = True
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collator.collate)

    val = get_autsl('validation')
    val.is_train = False
    val_loader = DataLoader(val, batch_size=args.batch_size, collate_fn=collator.collate)

    model = PoseSequenceClassification()
    # model = PoseSequenceClassification.load_from_checkpoint(
    #   "/home/nlp/amit/sign-language/sign-language-recognition/pose/models/27osn0ga/weights.ckpt.ckpt")

    callbacks = []

    if logger is not None:
        os.makedirs("models", exist_ok=True)
        os.makedirs("models/" + logger.experiment.id + "/", exist_ok=True)

        callbacks.append(ModelCheckpoint(
            filepath="models/" + logger.experiment.id + "/weights.ckpt",
            verbose=True,
            save_top_k=1,
            monitor='validation_acc_epoch',
            mode='max'
        ))

    trainer = pl.Trainer(
        max_epochs=50,
        logger=logger,
        callbacks=callbacks,
        gpus=args.gpus)

    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

    test = get_autsl('test')
    test.is_train = False
    test_loader = DataLoader(test, batch_size=args.batch_size, collate_fn=collator.collate)
    trainer.test(model, test_loader)
