import os
from os import path
from random import shuffle
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

GRID_SIZE = 1  # 6
FRAME_SIZE = 256
CHANNELS = 3
IMG_SIZE = FRAME_SIZE * GRID_SIZE


class VideoClassificationDataset(Dataset):
  def __init__(self, data: List[Tuple[str, Tuple[torch.Tensor, torch.Tensor]]]):
    self.data = data

  @staticmethod
  def from_tfds(tf_dataset):
    data = [
      (_id.numpy(),
       (torch.tensor(x.numpy(), dtype=torch.float32), torch.tensor(y.numpy(), dtype=torch.long)))
      for _id, x, y in tqdm(tf_dataset)
    ]

    _id, (x, y) = data[0]
    print("_id", _id)
    print("x", x.shape)
    print("y", y, y.shape)
    return VideoClassificationDataset(data)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    return self.data[index][1]


def get_mock_datum():
  frames = [np.full(shape=(FRAME_SIZE, FRAME_SIZE, CHANNELS), fill_value=i) for i in
            range(0, 255, int(255 / (GRID_SIZE * GRID_SIZE)))]
  video = np.stack(frames[:-1], axis=0)

  grid = video.reshape((GRID_SIZE, GRID_SIZE, FRAME_SIZE, FRAME_SIZE, CHANNELS))
  grid_frame = grid.swapaxes(1, 2).reshape(IMG_SIZE, IMG_SIZE, CHANNELS)

  return grid_frame.T


def get_mock_dataset(num_samples=1000):
  mock_datum_1 = torch.tensor(get_mock_datum(), dtype=torch.float32)
  mock_datum_2 = mock_datum_1.transpose(1, 2)

  cv2.imwrite("test1.png", mock_datum_1.numpy().T)
  cv2.imwrite("test2.png", mock_datum_2.numpy().T)

  data = [(mock_datum_1, torch.tensor(1)), (mock_datum_2, torch.tensor(0))] * (num_samples // 2)
  shuffle(data)
  return VideoClassificationDataset(data)


def get_autsl():
  cache_file = path.join(path.dirname(path.realpath(__file__)), "data.pt")
  print("Cache file", cache_file)
  if os.path.isfile(cache_file):
    return torch.load(cache_file)

  # noinspection PyUnresolvedReferences
  import sign_language_datasets.datasets
  from sign_language_datasets.datasets import SignDatasetConfig

  import tensorflow_datasets as tfds
  import tensorflow as tf

  config = SignDatasetConfig(name="256x256:10", include_video=True, fps=10, resolution=(256, 256))

  (all_set, test_set) = tfds.load(
    'autsl',
    builder_kwargs=dict(config=config, train_decryption_key="", valid_decryption_key=""),
    split=['train', 'validation'],
    shuffle_files=False,
    as_supervised=False,
  )

  def single_frame_extractor(datum):
    x = datum["video"]
    y = datum["gloss_id"]
    return datum["id"], tf.transpose(x[len(x) // 2]), y

  training_set = all_set.filter(
    lambda d: d["signer"] != 0 and d["signer"] != 12 and d["signer"] != 21 and d["signer"] != 22)
  validation_set = all_set.filter(
    lambda d: d["signer"] == 0 or d["signer"] == 12 or d["signer"] == 21 or d["signer"] == 22)

  validation_set = VideoClassificationDataset.from_tfds(validation_set.map(single_frame_extractor))
  test_set = VideoClassificationDataset.from_tfds(test_set.map(single_frame_extractor))
  training_set = VideoClassificationDataset.from_tfds(training_set.map(single_frame_extractor))

  res = training_set, validation_set, test_set
  torch.save(res, cache_file)
  return res
  #
  # sample = None
  # for datum in dataset["train"]:
  #     if datum["video"].shape[0] == 34:
  #         sample = datum
  #         break
  #
  # padding = tf.zeros((36 - sample["video"].shape[0], 256, 256, 3), dtype=tf.uint8)
  # video = tf.concat([sample["video"], padding], axis=0).numpy()
  # print(video.shape)
  # grid = video.reshape((GRID_SIZE, GRID_SIZE, FRAME_SIZE, FRAME_SIZE, CHANNELS))
  # grid_frame = grid.swapaxes(1, 2).reshape(IMG_SIZE, IMG_SIZE, CHANNELS)
  # print(grid_frame.shape)
  # cv2.imwrite("grid.png", cv2.cvtColor(grid_frame, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
  get_autsl()
  # get_mock_dataset()
