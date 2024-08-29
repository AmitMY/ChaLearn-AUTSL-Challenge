#!/bin/bash

source /home/gsantm/store/environment/pose_anonymization/bin/activate

export TFDS_DATA_DIR="/home/gsantm/store/data/tensorflow_datasets"
export CHECKPOINT_DIR="/home/gsantm/store/pose_anonymization_experiments/models"
export CHECKPOINT_NAME="epoch=9-step=550-v1"
export OUT_PATH="/home/gsantm/store/pose_anonymization_experiments/results"
export APPEARANCES_FILE="/home/gsantm/store/pose_anonymization_experiments/poses_files/1k.pose" # 'from_splis_signers', 'all_splits_appearances', '/home/gsantm/store/pose_anonymization_experiments/poses_files/10k.pose'

cd /home/gsantm/repositories/ChaLearn-AUTSL-Challenge

python pred.py \
    --no_wandb True \
    --checkpoint_dir $CHECKPOINT_DIR \
    --encoder transformer \
    --transfer_appearance False \
    --test_appearances_ids 123 456 789 101 112 \
    --appearances $APPEARANCES_FILE \
    --pretrained_checkpoint "${CHECKPOINT_DIR}/${CHECKPOINT_NAME}.ckpt" \
    --predictions_output "${OUT_PATH}/${CHECKPOINT_NAME}.csv"

    #--anonymize True \
    #
