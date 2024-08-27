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

def extract_subdictionary(original_dict, fraction=0.1):
    if fraction <= 0:
        return original_dict
    num_to_extract = int(len(original_dict) * fraction)
    keys_to_extract = random.sample(list(original_dict.keys()), num_to_extract)
    new_dict = {key: original_dict[key] for key in keys_to_extract}
    for key in keys_to_extract:
        del original_dict[key]
    return new_dict

if __name__ == '__main__':
    if not args.no_wandb:
        logger = WandbLogger(project="autsl", log_model=False, offline=False)
        if logger.experiment.sweep_id is None:
            logger.log_hyperparams(args)
    else:
        logger = None

    collator = ZeroPadCollator()

    print("#######################################################")
    print(f"args.appearances ({type(args.appearances)}): {args.appearances}")
    print(f"os.path.isfile(value): {os.path.isfile(args.appearances)}")
    print("#######################################################")

    train = get_autsl('train')
    val = get_autsl('validation')

    train.is_train = True
    val.is_train = False

    if args.anonymize:

        train.anonymize = True
        train.transfer_appearance = False
        val.anonymize = True
        val.transfer_appearance = False
    
    if args.transfer_appearance:

        train.anonymize = False
        train.transfer_appearance = True
        val.anonymize = False
        val.transfer_appearance = True

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
            val_apperances_dict = extract_subdictionary(apperances_dict, fraction=args.validation_signers_fraction)
            train.signers_poses = apperances_dict
            val.signers_poses = val_apperances_dict

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collator.collate)
    val_loader = DataLoader(val, batch_size=args.batch_size, collate_fn=collator.collate)

    model = PoseSequenceClassification()
    # model = PoseSequenceClassification.load_from_checkpoint(
    #   "/home/nlp/amit/sign-language/sign-language-recognition/pose/models/27osn0ga/weights.ckpt.ckpt")

    callbacks = []

    callbacks.append(ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        verbose=True,
        save_top_k=3,
        monitor='validation_acc_epoch',
        mode='max'
    ))

    num_gpus = torch.cuda.device_count()
    devices = num_gpus if num_gpus > 0 else 1

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
