
import os
import random
from pose_format import Pose

from collections import Counter
from zipfile import ZipFile

import torch
import numpy as np
from scipy.stats import mode
from torch.utils.data import DataLoader
from tqdm import tqdm

from args import args
from data import get_autsl, ZeroPadCollator
from model import PoseSequenceClassification
from pose_anonymization.appearance import remove_appearance, transfer_appearance

def augmented_evaluation(model, test, collator, gold_values):
  if os.path.isfile(args.appearances):
      with open(args.appearances, "rb") as f:
          appearances_file = Pose.read(f.read())
      apperances_dict = {}
      for frame_index in range(appearances_file.body.data.shape[0]):
          apperances_dict[frame_index] = Pose(appearances_file.header, appearances_file.body[frame_index:frame_index+1])
      test.signers_poses = apperances_dict

  _predictions = {datum["id"]: [] for datum in test.data}

  with torch.no_grad():
    for index in tqdm(range(len(test))):
      batch = []
      for signer_id in args.test_appearances_ids:
        datum = test.data[index]
        pose = datum["pose"]
        target_appearance_pose = test.signers_poses[signer_id]
        pose = transfer_appearance(pose=pose, appearance_pose=target_appearance_pose)

        torch_body_data = pose.body.torch().data.cuda()

        sample = {
              "id": datum["id"],
              "signer": datum["signer"],
              "pose": torch_body_data,
              "length": torch.ones((len(torch_body_data)), dtype=torch.bool),
              "label": datum["label"]
          }

        batch.append(sample)
      batch = collator.collate(batch)

      y_hats = model.pred(batch).cpu().numpy()
      for i, y_hat in enumerate(y_hats):
          datum_id = batch['id'][i]
          _predictions[datum_id].append(y_hat)

  predictions = {}
  for datum_id, preds in _predictions.items():
      most_common = mode(preds)
      predictions[datum_id] = most_common.mode[0]

  return predictions


def evaluate(model, test, collator, gold_values):

  if args.anonymize:
    test.anonymize = True
  
  test_loader = DataLoader(test, batch_size=args.batch_size, collate_fn=collator.collate)

  predictions = {}

  with torch.no_grad():
    for batch in tqdm(test_loader):
      batch["pose"] = batch["pose"].cuda()

      y_hats = model.pred(batch).cpu().numpy()

      for _id, y_hat in zip(batch["id"], y_hats):
        predictions[_id] = y_hat

  return predictions


if __name__ == '__main__':
  model = PoseSequenceClassification.load_from_checkpoint(checkpoint_path=args.pretrained_checkpoint, heads=args.encoder_heads, depth=args.encoder_depth)
  model = model.cuda()
  test = get_autsl('test')
  test.is_train = False
  gold_values = {datum["id"]: datum["label"] for datum in test.data}
  collator = ZeroPadCollator()

  if args.transfer_appearance:
    predictions = augmented_evaluation(model, test, collator, gold_values)
  else:
    predictions = evaluate(model, test, collator, gold_values)

  correct = len([1 for _id, y_hat in predictions.items() if y_hat == gold_values[_id]])

  print("Accuracy", correct / len(predictions))

  with open(args.predictions_output, "w") as f:
    for _id, y_hat in predictions.items():
      f.write(_id.decode('utf-8') + "," + str(y_hat) + "\n")

  with ZipFile(f'{args.predictions_output[:-4]}.zip', 'w') as zipObj:
    zipObj.write(args.predictions_output)
