import os
import torch
import random
from pose_format import Pose

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

# noinspection PyUnresolvedReferences
from args import args
from data import get_autsl, ZeroPadCollator
from model import PoseSequenceClassification

if __name__ == '__main__':
    if not args.no_wandb:
        logger = WandbLogger(project="autsl", log_model=False, offline=False)
        if logger.experiment.sweep_id is None:
            logger.log_hyperparams(args)
    else:
        logger = None

    collator = ZeroPadCollator()

    train = get_autsl(
        split='train',
        anonymize=args.anonymize,
        transfer_appearance=args.transfer_appearance
        )

    val = get_autsl(
        split='validation',
        anonymize=args.anonymize,
        transfer_appearance=args.transfer_appearance
        )

    train.is_train = True
    val.is_train = False
    
    if args.transfer_appearance:

        if args.appearances == "all_splits_appearances":
            all_appearances_dict = {**train.signers_poses, **val.signers_poses}
            train.signers_poses = all_appearances_dict
            val.signers_poses = all_appearances_dict

        elif os.path.isfile(args.appearances):
            with open(args.appearances, "rb") as f:
                appearances_file = Pose.read(f.read())
            apperances_dict = {}
            for frame_index in range(appearances_file.body.data.shape[0]):
                apperances_dict[frame_index] = Pose(appearances_file.header, appearances_file.body[frame_index:frame_index+1])
            train.signers_poses = apperances_dict
            val.signers_poses = apperances_dict

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collator.collate)
    val_loader = DataLoader(val, batch_size=args.batch_size, collate_fn=collator.collate)

    model = PoseSequenceClassification()

    callbacks = []

    callbacks.append(ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        verbose=True,
        save_top_k=3,
        monitor='validation_acc_epoch',
        mode='max'
    ))

    num_gpus = torch.cuda.device_count()
    devices = max(num_gpus, 1)

    trainer = pl.Trainer(
        max_epochs=50,
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",
        devices=devices)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    test = get_autsl('test')
    test.is_train = False
    test_loader = DataLoader(test, batch_size=args.batch_size, collate_fn=collator.collate)
    trainer.test(model, test_loader)
