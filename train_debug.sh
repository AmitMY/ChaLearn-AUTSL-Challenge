#!/bin/bash

source /home/gsantm/store/environment/pose_anonymization/bin/activate

export TFDS_DATA_DIR="/home/gsantm/store/data/tensorflow_datasets"
export CHECKPOINT_DIR="/home/gsantm/store/pose_anonymization_experiments"
export APPEARANCES_FILE="/home/gsantm/store/pose_anonymization_experiments/poses_files/10k.pose" # 'from_splis_signers', 'all_splits_appearances', '/home/gsantm/store/pose_anonymization_experiments/poses_files/10k.pose'

cd /home/gsantm/repositories/ChaLearn-AUTSL-Challenge

python train_debug.py \
    --no_wandb True \
    --checkpoint_dir $CHECKPOINT_DIR \
    --encoder transformer \
    --transfer_appearance True \
    --appearances $APPEARANCES_FILE

    #--anonymize True \
