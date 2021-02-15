import os
import random
from os import path
from typing import List

import torch
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from torch.utils.data import Dataset
from tqdm import tqdm

from pose.args import args, POSE_HEADER


class PoseClassificationDataset(Dataset):
  def __init__(self, data: List, is_train=False):
    self.data = data
    self.is_train = is_train

  @staticmethod
  def from_tfds(tf_dataset, **kwargs):
    info = POSE_HEADER.normalization_info(
      p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
      p2=("POSE_LANDMARKS", "LEFT_SHOULDER")
    )

    data = []

    for datum in tqdm(tf_dataset):
      body = NumPyPoseBody(30, datum["pose"]["data"], datum["pose"]["conf"])
      pose = Pose(POSE_HEADER, body).get_components(args.pose_components).normalize(info)
      new_fps = (args.seq_size / len(pose.body.data)) * 30
      pose = pose.interpolate(new_fps=new_fps, kind='linear')

      data.append({
        "id": datum["id"].numpy(),
        "signer": datum["signer"].numpy(),
        "pose": pose,
        "label": torch.tensor(datum["gloss_id"].numpy(), dtype=torch.long)
      })

    return PoseClassificationDataset(data, **kwargs)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    datum = self.data[index]
    pose = datum["pose"]

    # # Setup pose representation
    # if self.rep is None:
    #   self.rep = TorchPoseRepresentation(pose.header, [AngleRepresentation(), DistanceRepresentation()])

    # Only use requested dimensions
    if pose.body.data.shape[-1] != args.pose_dims:
      pose.body.data = pose.body.data.T[:args.pose_dims].T

    if self.is_train:
      # # x direction flip 50% of the time for left handed / two handed signs
      # if random.randint(1, 10) <= 5:
      #   pose = pose.flip(axis=0)

      pose = pose.augment2d(rotation_std=args.rotation_std, scale_std=args.scale_std, shear_std=args.shear_std)

    x = torch.tensor(pose.body.data.data, dtype=torch.float)

    return datum["id"], datum["signer"], x, datum["label"]


def get_autsl(split: str):
  cache_file = path.join(path.dirname(path.realpath(__file__)), split + "_data.pt")
  print("Cache file", cache_file)
  if os.path.isfile(cache_file):
    return torch.load(cache_file)

  # noinspection PyUnresolvedReferences
  import sign_language_datasets.datasets
  from sign_language_datasets.datasets.config import SignDatasetConfig

  import tensorflow_datasets as tfds

  config = SignDatasetConfig(name="poses", include_video=False, include_pose="holistic")

  data_set = tfds.load(
    'autsl',
    builder_kwargs=dict(config=config, train_decryption_key="", valid_decryption_key=""),
    split=split,
    shuffle_files=False,
    as_supervised=False,
  )

  data_set = PoseClassificationDataset.from_tfds(data_set)

  torch.save(data_set, cache_file)
  return data_set


def split_train_dataset(dataset: PoseClassificationDataset, ids):
  ids = set(ids)
  train_data = [d for d in dataset.data if d["signer"] not in ids]
  valid_data = [d for d in dataset.data if d["signer"] in ids]

  return PoseClassificationDataset(train_data, is_train=True), PoseClassificationDataset(valid_data, is_train=False)


if __name__ == "__main__":
  get_autsl()
